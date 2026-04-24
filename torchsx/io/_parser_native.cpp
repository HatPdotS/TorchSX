// Native C++ parser for CrystFEL .stream files.
//
// Matches the Python state machine in _parser.py exactly. All numeric
// output tensors are written directly via data_ptr() — no Python-level
// indexing, no numpy view.
//
// The file is memory-mapped for maximum read speed. Each worker handles a
// disjoint byte range [start, end); ranges are expected to be chunk-aligned
// so workers never split a chunk.

#include <torch/extension.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

// Buffered sequential reader over a file byte range [start, end).
//
// Holds a fixed-size heap buffer (default 4 MiB). Lines are returned as
// ``std::string_view`` pointing into the buffer; the view is valid only
// until the *next* ``read_line`` call (which may shift and refill). Lines
// longer than the buffer are supported — the buffer grows to fit.
//
// Unlike mmap-the-whole-file, the RSS footprint is bounded to the buffer
// size plus any OS page-cache activity for the current read stride.
struct BufferedReader {
    int fd = -1;
    int64_t file_end = 0;  // do not read past this file offset
    int64_t file_pos = 0;  // next file offset to read from
    std::vector<char> buf;
    size_t head = 0;
    size_t tail = 0;
    std::vector<char> line_overflow;  // used when a line exceeds buf capacity

    BufferedReader(const std::string& path, int64_t start, int64_t end,
                   size_t cap = 4 << 20)
        : fd(::open(path.c_str(), O_RDONLY)),
          file_end(end),
          file_pos(start),
          buf(cap) {
        if (fd < 0) {
            throw std::runtime_error(
                "open failed: " + path + ": " + std::strerror(errno));
        }
        if (::lseek(fd, start, SEEK_SET) != start) {
            ::close(fd);
            throw std::runtime_error("lseek failed");
        }
        ::posix_fadvise(fd, start, std::max<int64_t>(0, end - start),
                        POSIX_FADV_SEQUENTIAL);
    }
    ~BufferedReader() {
        if (fd >= 0) ::close(fd);
    }
    BufferedReader(const BufferedReader&) = delete;
    BufferedReader& operator=(const BufferedReader&) = delete;

    // Refill `buf` from file. Returns false if EOF / end reached and nothing
    // was read.
    bool refill() {
        if (file_pos >= file_end) return false;
        int64_t want = std::min<int64_t>(
            static_cast<int64_t>(buf.size() - tail), file_end - file_pos);
        while (want > 0) {
            ssize_t got = ::read(fd, buf.data() + tail, want);
            if (got == 0) break;
            if (got < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error(std::string("read failed: ") +
                                         std::strerror(errno));
            }
            tail += static_cast<size_t>(got);
            file_pos += got;
            want -= got;
        }
        return tail > 0;
    }

    // Return next line as view (without trailing \n/\r). Returns false on
    // EOF. View valid until next call.
    bool read_line(std::string_view& out) {
        if (head == tail && !refill()) return false;
        for (;;) {
            // scan for newline within available bytes
            const char* base = buf.data() + head;
            size_t avail = tail - head;
            const void* nl = std::memchr(base, '\n', avail);
            if (nl) {
                const char* e = static_cast<const char*>(nl);
                size_t len = static_cast<size_t>(e - base);
                // strip trailing \r
                size_t strip = (len > 0 && base[len - 1] == '\r') ? 1 : 0;
                out = std::string_view(base, len - strip);
                head += len + 1;  // consume \n
                return true;
            }
            // No newline in current buffer. Either:
            //   (a) buffer has free tail space → refill and keep scanning.
            //   (b) buffer is full (line too long) → grow via overflow vec.
            //   (c) EOF: return the remainder as a line (no trailing \n).
            if (tail < buf.size()) {
                if (!refill()) {
                    if (head < tail) {
                        size_t len = tail - head;
                        size_t strip = (len > 0 && buf[tail - 1] == '\r') ? 1 : 0;
                        out = std::string_view(buf.data() + head, len - strip);
                        head = tail;
                        return true;
                    }
                    return false;
                }
                continue;
            }
            // Buffer is full. Shift remaining bytes to front, then refill.
            if (head > 0) {
                std::memmove(buf.data(), buf.data() + head, tail - head);
                tail -= head;
                head = 0;
                if (!refill()) {
                    // EOF with a final unterminated line in the buffer
                    if (tail > 0) {
                        size_t strip = (buf[tail - 1] == '\r') ? 1 : 0;
                        out = std::string_view(buf.data(), tail - strip);
                        head = tail;
                        return true;
                    }
                    return false;
                }
                continue;
            }
            // Line truly exceeds buffer capacity — grow.
            buf.resize(buf.size() * 2);
        }
    }

    int64_t tell() const { return file_pos - static_cast<int64_t>(tail - head); }
};

// Skip whitespace (space/tab only, not newline).
inline void skip_sp(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t')) ++p;
}

// Parse a signed int in base 10. Advances p. Returns true on success.
inline bool parse_int(const char*& p, const char* end, long& out) {
    skip_sp(p, end);
    if (p >= end) return false;
    char* e = nullptr;
    long v = std::strtol(p, &e, 10);
    if (e == p) return false;
    out = v;
    p = e;
    return true;
}

// All floats are parsed as double — matching Python's ``float()`` — then
// cast to float32 only on final tensor store. Using ``strtof`` would round
// to float32 before the subsequent arithmetic (×10 for cell, /10 for
// reciprocals), which produces bit-different results vs the Python path.
inline bool parse_double(const char*& p, const char* end, double& out) {
    skip_sp(p, end);
    if (p >= end) return false;
    char* e = nullptr;
    double v = std::strtod(p, &e);
    if (e == p) return false;
    out = v;
    p = e;
    return true;
}

// Parse a whitespace-terminated token as string_view into [p, next_ws).
inline std::string_view parse_token(const char*& p, const char* end) {
    skip_sp(p, end);
    const char* s = p;
    while (p < end && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
        ++p;
    return {s, static_cast<size_t>(p - s)};
}

inline bool starts_with(std::string_view sv, const char* prefix) {
    size_t n = std::strlen(prefix);
    return sv.size() >= n && std::memcmp(sv.data(), prefix, n) == 0;
}

inline bool find_substr(std::string_view sv, const char* needle) {
    return sv.find(needle) != std::string_view::npos;
}

inline std::string_view strip(std::string_view s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t')) --e;
    return s.substr(b, e - b);
}

// Parse a key=val line, returning key and the first whitespace-separated
// token of val. Matches the Python parser's behavior.
inline bool parse_kv(std::string_view line, std::string_view& key,
                     std::string_view& val_first) {
    size_t eq = line.find('=');
    if (eq == std::string_view::npos) return false;
    key = strip(line.substr(0, eq));
    std::string_view v = strip(line.substr(eq + 1));
    size_t ws = v.find_first_of(" \t");
    if (ws == std::string_view::npos) {
        val_first = v;
    } else {
        val_first = v.substr(0, ws);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Counting pass (mirrors _count_byte_range in Python).
// ---------------------------------------------------------------------------

py::tuple count_byte_range(const std::string& path, int64_t start, int64_t end) {
    BufferedReader r(path, start, end);
    int64_t n_obs = 0, n_crys = 0, n_imgs = 0;
    bool in_refl = false;
    std::string_view line;
    while (r.read_line(line)) {
        if (in_refl) {
            if (starts_with(line, "End of reflections")) {
                in_refl = false;
                continue;
            }
            int words = 0;
            const char* q = line.data();
            const char* qe = q + line.size();
            while (q < qe) {
                while (q < qe && (*q == ' ' || *q == '\t')) ++q;
                if (q >= qe) break;
                ++words;
                if (words >= 10) break;
                while (q < qe && *q != ' ' && *q != '\t') ++q;
            }
            if (words >= 10) ++n_obs;
        } else {
            if (starts_with(line, "Image filename:")) {
                ++n_imgs;
            } else if (starts_with(line, "--- Begin crystal")) {
                ++n_crys;
            } else if (starts_with(line, "Reflections measured after indexing")) {
                in_refl = true;
                r.read_line(line);  // skip header line
            }
        }
    }
    return py::make_tuple(n_obs, n_crys, n_imgs);
}

// ---------------------------------------------------------------------------
// Find chunk boundary at or after target.
// ---------------------------------------------------------------------------

int64_t find_chunk_at_or_after(const std::string& path, int64_t target) {
    struct stat st;
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("open failed: " + path + ": " + std::strerror(errno));
    }
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("fstat failed");
    }
    int64_t size = static_cast<int64_t>(st.st_size);
    ::close(fd);

    BufferedReader r(path, std::min<int64_t>(target, size), size);
    std::string_view line;
    if (target > 0) {
        // Consume the (possibly partial) line we landed in.
        int64_t before = r.tell();
        if (!r.read_line(line)) return size;
    }
    while (true) {
        int64_t line_start = r.tell();
        if (!r.read_line(line)) return size;
        if (starts_with(line, "----- Begin chunk -----")) {
            return line_start;
        }
    }
}

// ---------------------------------------------------------------------------
// Parse pass: write directly into pre-allocated tensors at given offsets.
// ---------------------------------------------------------------------------

struct TensorPtrs {
    int16_t* hkl;
    float* I;
    float* sigma;
    float* peak;
    float* background;
    float* fs;
    float* ss;
    int16_t* panel_id;
    int32_t* crystal_id;

    float* cell;
    float* astar;
    float* bstar;
    float* cstar;
    float* profile_rad;
    float* resolution;
    int32_t* n_refl;
    int32_t* image_id;
    int32_t* crystal_in_image;

    float* beam_energy;
    float* det_shift_x;
    float* det_shift_y;
    float* camera_length;
    int32_t* n_peaks;
};

// Returns (n_obs_seg, n_crys_seg, n_imgs_seg, filenames, events).
py::tuple parse_into_shared(
    const std::string& path,
    int64_t start,
    int64_t end,
    const std::unordered_map<std::string, int>& panel_name_to_id,
    torch::Tensor hkl,
    torch::Tensor I,
    torch::Tensor sigma,
    torch::Tensor peak,
    torch::Tensor background,
    torch::Tensor fs,
    torch::Tensor ss,
    torch::Tensor panel_id,
    torch::Tensor crystal_id,
    torch::Tensor cell,
    torch::Tensor astar,
    torch::Tensor bstar,
    torch::Tensor cstar,
    torch::Tensor profile_rad,
    torch::Tensor resolution,
    torch::Tensor n_refl,
    torch::Tensor image_id,
    torch::Tensor crystal_in_image,
    torch::Tensor beam_energy,
    torch::Tensor det_shift_x,
    torch::Tensor det_shift_y,
    torch::Tensor camera_length,
    torch::Tensor n_peaks,
    int64_t obs_offset,
    int64_t crys_offset,
    int64_t img_offset
) {
    TensorPtrs T;
    T.hkl              = hkl.data_ptr<int16_t>();
    T.I                = I.data_ptr<float>();
    T.sigma            = sigma.data_ptr<float>();
    T.peak             = peak.data_ptr<float>();
    T.background       = background.data_ptr<float>();
    T.fs               = fs.data_ptr<float>();
    T.ss               = ss.data_ptr<float>();
    T.panel_id         = panel_id.data_ptr<int16_t>();
    T.crystal_id       = crystal_id.data_ptr<int32_t>();
    T.cell             = cell.data_ptr<float>();
    T.astar            = astar.data_ptr<float>();
    T.bstar            = bstar.data_ptr<float>();
    T.cstar            = cstar.data_ptr<float>();
    T.profile_rad      = profile_rad.data_ptr<float>();
    T.resolution       = resolution.data_ptr<float>();
    T.n_refl           = n_refl.data_ptr<int32_t>();
    T.image_id         = image_id.data_ptr<int32_t>();
    T.crystal_in_image = crystal_in_image.data_ptr<int32_t>();
    T.beam_energy      = beam_energy.data_ptr<float>();
    T.det_shift_x      = det_shift_x.data_ptr<float>();
    T.det_shift_y      = det_shift_y.data_ptr<float>();
    T.camera_length    = camera_length.data_ptr<float>();
    T.n_peaks          = n_peaks.data_ptr<int32_t>();

    BufferedReader r(path, start, end);
    std::string_view line;

    std::vector<std::string> filenames;
    std::vector<std::string> events;

    int64_t crystal_idx = crys_offset;
    int64_t image_idx   = img_offset;
    int64_t n_obs       = obs_offset;

    while (r.read_line(line)) {
        if (!starts_with(line, "Image filename:")) continue;

        // ---- new chunk / image ----
        filenames.emplace_back(strip(line.substr(15)));

        if (!r.read_line(line)) break;
        if (starts_with(line, "Event:")) {
            events.emplace_back(strip(line.substr(6)));
        } else {
            events.emplace_back("");
        }

        // ---- chunk key/value pairs ----
        double photon_energy_eV = 0.0;
        int32_t num_peaks = 0;
        double avg_clen = 0.0;
        while (r.read_line(line)) {
            if (starts_with(line, "Peaks from peak search")) break;
            if (starts_with(line, "----- End chunk -----")) break;
            std::string_view key, val;
            if (parse_kv(line, key, val)) {
                if (key == "photon_energy_eV") {
                    const char* vp = val.data();
                    parse_double(vp, val.data() + val.size(), photon_energy_eV);
                } else if (key == "num_peaks") {
                    const char* vp = val.data();
                    long v = 0;
                    if (parse_int(vp, val.data() + val.size(), v)) num_peaks = static_cast<int32_t>(v);
                } else if (key == "average_camera_length") {
                    const char* vp = val.data();
                    parse_double(vp, val.data() + val.size(), avg_clen);
                }
            }
        }

        int64_t current_image_id = image_idx;
        T.beam_energy[current_image_id]   = static_cast<float>(photon_energy_eV);
        T.n_peaks[current_image_id]       = num_peaks;
        T.det_shift_x[current_image_id]   = 0.0f;
        T.det_shift_y[current_image_id]   = 0.0f;
        T.camera_length[current_image_id] = static_cast<float>(avg_clen);
        ++image_idx;
        int32_t crystal_in_image_counter = 0;

        // Skip peak list
        while (r.read_line(line)) {
            if (starts_with(line, "End of peak list")) break;
        }

        // Parse crystals in this chunk
        bool chunk_done = false;
        while (!chunk_done && r.read_line(line)) {
            if (starts_with(line, "----- End chunk -----")) break;
            if (!starts_with(line, "--- Begin crystal")) continue;

            // Cell parameters (nm → Å on a,b,c; α,β,γ degrees)
            if (!r.read_line(line)) { chunk_done = true; break; }
            std::string_view cell_line = line;
            {
                double a[6] = {0, 0, 0, 0, 0, 0};
                const char* cp = cell_line.data();
                const char* ce = cp + cell_line.size();
                int got = 0;
                while (got < 6 && cp < ce) {
                    skip_sp(cp, ce);
                    char* e = nullptr;
                    double v = std::strtod(cp, &e);
                    if (e == cp) {
                        // not a float: skip to next space
                        while (cp < ce && *cp != ' ' && *cp != '\t') ++cp;
                    } else {
                        a[got++] = v;
                        cp = e;
                    }
                }
                float* row = T.cell + crystal_idx * 6;
                row[0] = static_cast<float>(a[0] * 10.0);
                row[1] = static_cast<float>(a[1] * 10.0);
                row[2] = static_cast<float>(a[2] * 10.0);
                row[3] = static_cast<float>(a[3]);
                row[4] = static_cast<float>(a[4]);
                row[5] = static_cast<float>(a[5]);
            }

            // astar / bstar / cstar (nm⁻¹ → Å⁻¹)
            bool got_vecs = true;
            for (int vec_idx = 0; vec_idx < 3; ++vec_idx) {
                if (!r.read_line(line)) { got_vecs = false; break; }
                // format: "astar = -0.513 -0.297 -0.420 nm^-1"
                size_t eq = line.find('=');
                double v[3] = {0, 0, 0};
                int got = 0;
                if (eq != std::string_view::npos) {
                    const char* cp = line.data() + eq + 1;
                    const char* ce = line.data() + line.size();
                    while (got < 3 && cp < ce) {
                        skip_sp(cp, ce);
                        char* e = nullptr;
                        double val = std::strtod(cp, &e);
                        if (e == cp) {
                            while (cp < ce && *cp != ' ' && *cp != '\t') ++cp;
                        } else {
                            v[got++] = val;
                            cp = e;
                        }
                    }
                }
                float* ptr = nullptr;
                if (vec_idx == 0) ptr = T.astar + crystal_idx * 3;
                else if (vec_idx == 1) ptr = T.bstar + crystal_idx * 3;
                else ptr = T.cstar + crystal_idx * 3;
                ptr[0] = static_cast<float>(v[0] / 10.0);
                ptr[1] = static_cast<float>(v[1] / 10.0);
                ptr[2] = static_cast<float>(v[2] / 10.0);
            }
            if (!got_vecs) { chunk_done = true; break; }

            // Crystal key/value pairs
            double profile_rad_nm = 0.04;
            double res_nm = 0.0;
            double det_shift_x_mm = 0.0;
            double det_shift_y_mm = 0.0;
            bool got_det_shift = false;
            while (r.read_line(line)) {
                if (starts_with(line, "Reflections measured after indexing")) break;
                size_t eq = line.find('=');
                if (eq == std::string_view::npos) continue;
                std::string_view key = strip(line.substr(0, eq));
                std::string_view val = strip(line.substr(eq + 1));
                if (key.find("det_shift") != std::string_view::npos
                    && val.find("y =") != std::string_view::npos) {
                    size_t ypos = val.find("y =");
                    std::string_view xpart = strip(val.substr(0, ypos));
                    std::string_view ypart = strip(val.substr(ypos + 3));
                    if (ypart.size() >= 2 && ypart.substr(ypart.size() - 2) == "mm") {
                        ypart = strip(ypart.substr(0, ypart.size() - 2));
                    }
                    double xv = 0, yv = 0;
                    const char* xp = xpart.data();
                    parse_double(xp, xpart.data() + xpart.size(), xv);
                    const char* yp = ypart.data();
                    parse_double(yp, ypart.data() + ypart.size(), yv);
                    det_shift_x_mm = xv;
                    det_shift_y_mm = yv;
                    got_det_shift = true;
                } else if (key == "profile_radius") {
                    const char* vp = val.data();
                    parse_double(vp, val.data() + val.size(), profile_rad_nm);
                } else if (key == "diffraction_resolution_limit") {
                    const char* vp = val.data();
                    parse_double(vp, val.data() + val.size(), res_nm);
                }
            }

            T.profile_rad[crystal_idx] = static_cast<float>(profile_rad_nm / 10.0);
            T.resolution[crystal_idx]  = static_cast<float>((res_nm > 0.0) ? (10.0 / res_nm) : 999.0);
            if (got_det_shift) {
                T.det_shift_x[current_image_id] = static_cast<float>(det_shift_x_mm);
                T.det_shift_y[current_image_id] = static_cast<float>(det_shift_y_mm);
            }

            // Skip reflection-table header line
            if (!r.read_line(line)) { chunk_done = true; break; }

            // Reflections
            int32_t n_refl_count = 0;
            while (r.read_line(line)) {
                if (starts_with(line, "End of reflections")) break;

                const char* cp = line.data();
                const char* ce = cp + line.size();
                long h, k, l;
                double I_val, sigma_val, peak_val, bg_val, fs_val, ss_val;
                if (!parse_int(cp, ce, h)) continue;
                if (!parse_int(cp, ce, k)) continue;
                if (!parse_int(cp, ce, l)) continue;
                if (!parse_double(cp, ce, I_val)) continue;
                if (!parse_double(cp, ce, sigma_val)) continue;
                if (!parse_double(cp, ce, peak_val)) continue;
                if (!parse_double(cp, ce, bg_val)) continue;
                if (!parse_double(cp, ce, fs_val)) continue;
                if (!parse_double(cp, ce, ss_val)) continue;
                std::string_view panel_name = parse_token(cp, ce);
                if (panel_name.empty()) continue;

                int pid = 0;
                {
                    // unordered_map<std::string, int>::find wants std::string;
                    // build a small stack string to avoid alloc when possible.
                    std::string key(panel_name);
                    auto it = panel_name_to_id.find(key);
                    if (it != panel_name_to_id.end()) pid = it->second;
                }

                int64_t r = n_obs;
                T.hkl[r * 3 + 0]   = static_cast<int16_t>(h);
                T.hkl[r * 3 + 1]   = static_cast<int16_t>(k);
                T.hkl[r * 3 + 2]   = static_cast<int16_t>(l);
                T.I[r]             = static_cast<float>(I_val);
                T.sigma[r]         = static_cast<float>(sigma_val);
                T.peak[r]          = static_cast<float>(peak_val);
                T.background[r]    = static_cast<float>(bg_val);
                T.fs[r]            = static_cast<float>(fs_val);
                T.ss[r]            = static_cast<float>(ss_val);
                T.panel_id[r]      = static_cast<int16_t>(pid);
                T.crystal_id[r]    = static_cast<int32_t>(crystal_idx);
                ++n_obs;
                ++n_refl_count;
            }

            T.n_refl[crystal_idx]           = n_refl_count;
            T.image_id[crystal_idx]         = static_cast<int32_t>(current_image_id);
            T.crystal_in_image[crystal_idx] = crystal_in_image_counter;
            ++crystal_in_image_counter;
            ++crystal_idx;

            // Skip to "--- End crystal"
            while (r.read_line(line)) {
                if (starts_with(line, "--- End crystal")) break;
            }
        }
    }

    return py::make_tuple(
        n_obs - obs_offset,
        crystal_idx - crys_offset,
        image_idx - img_offset,
        filenames,
        events
    );
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("count_byte_range", &count_byte_range,
          "Count (n_obs, n_crystals, n_images) in a byte range (mmap).");
    m.def("find_chunk_at_or_after", &find_chunk_at_or_after,
          "Find byte offset of first '----- Begin chunk -----' at or after target.");
    m.def("parse_into_shared", &parse_into_shared,
          "Parse a byte range, writing into pre-allocated shared tensors at the given offsets.");
}
