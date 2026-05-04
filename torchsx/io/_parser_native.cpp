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

#include <array>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
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
// Per-chunk indexing: emit (start, end, n_crys, n_obs) tuples.
//
// Single-threaded scan over a byte range, with optional early-exit when
// cumulative crystals reach `max_crystals`. The multi-threaded entry
// point splits the file into chunk-aligned ranges and merges results.
// ---------------------------------------------------------------------------

struct RangeResult {
    std::vector<std::array<int64_t, 4>> chunks;  // (start, end, n_crys, n_obs)
    bool has_pending = false;
    int64_t pending_start = 0;
    int64_t pending_n_crys = 0;
    int64_t pending_n_obs = 0;
};

// Scan [start, end) of `path`, populating `res`.
//
// Each completed chunk gets its end set to the byte offset of the next
// '----- Begin chunk -----'. The last chunk in the range — whose end lies
// outside [start, end) — is returned as `pending` (no end set); the caller
// stitches it to either the next worker's first chunk start or EOF.
//
// If `max_crystals > 0`, stops as soon as cumulative crystals reach the
// threshold; in that case `has_pending` is false because the threshold-
// crossing chunk was already finalized.
void scan_range_into(const std::string& path, int64_t start, int64_t end,
                     int64_t max_crystals, RangeResult& res) {
    BufferedReader r(path, start, end);
    int64_t current_start = -1;
    int64_t current_crys = 0;
    int64_t current_obs = 0;
    int64_t cum_crys = 0;
    bool in_refl = false;
    std::string_view line;

    while (true) {
        int64_t pos = r.tell();
        if (!r.read_line(line)) break;

        if (in_refl) {
            if (starts_with(line, "End of reflections")) {
                in_refl = false;
            } else {
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
                if (words >= 10) ++current_obs;
            }
            continue;
        }

        if (starts_with(line, "----- Begin chunk -----")) {
            if (current_start >= 0) {
                res.chunks.push_back({current_start, pos, current_crys, current_obs});
                cum_crys += current_crys;
                if (max_crystals > 0 && cum_crys >= max_crystals) {
                    res.has_pending = false;
                    return;
                }
            }
            current_start = pos;
            current_crys = 0;
            current_obs = 0;
        } else if (starts_with(line, "--- Begin crystal")) {
            ++current_crys;
        } else if (starts_with(line, "Reflections measured after indexing")) {
            in_refl = true;
            r.read_line(line);  // skip header
        }
    }

    if (current_start >= 0) {
        res.has_pending = true;
        res.pending_start = current_start;
        res.pending_n_crys = current_crys;
        res.pending_n_obs = current_obs;
    }
}

py::list index_chunks(const std::string& path, int n_threads, int64_t max_crystals) {
    // file size
    struct stat st;
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("open failed: " + path + ": " + std::strerror(errno));
    }
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("fstat failed");
    }
    int64_t file_size = static_cast<int64_t>(st.st_size);
    ::close(fd);

    int64_t header_end = find_chunk_at_or_after(path, 0);
    if (header_end >= file_size) {
        return py::list();
    }

    if (n_threads < 1) n_threads = 1;

    // Single-threaded path: also used for early-exit head mode, where
    // threading would just race past the threshold.
    if (n_threads == 1 || max_crystals > 0) {
        RangeResult res;
        {
            py::gil_scoped_release nogil;
            scan_range_into(path, header_end, file_size, max_crystals, res);
        }
        py::list out;
        for (auto& c : res.chunks) {
            out.append(py::make_tuple(c[0], c[1], c[2], c[3]));
        }
        if (res.has_pending) {
            out.append(py::make_tuple(res.pending_start, file_size,
                                      res.pending_n_crys, res.pending_n_obs));
        }
        return out;
    }

    // Multi-threaded: chunk-aligned byte-range split, one std::thread per range.
    std::vector<int64_t> starts;
    starts.push_back(header_end);
    for (int i = 1; i < n_threads; ++i) {
        int64_t target = header_end + (file_size - header_end) * i / n_threads;
        int64_t s = find_chunk_at_or_after(path, target);
        if (s > starts.back() && s < file_size) {
            starts.push_back(s);
        }
    }
    int actual_n = static_cast<int>(starts.size());
    std::vector<int64_t> ends(actual_n);
    for (int i = 0; i < actual_n - 1; ++i) ends[i] = starts[i + 1];
    ends[actual_n - 1] = file_size;

    std::vector<RangeResult> results(actual_n);
    std::vector<std::exception_ptr> errors(actual_n, nullptr);
    {
        py::gil_scoped_release nogil;
        std::vector<std::thread> threads;
        threads.reserve(actual_n);
        for (int i = 0; i < actual_n; ++i) {
            threads.emplace_back([&, i]() {
                try {
                    scan_range_into(path, starts[i], ends[i], -1, results[i]);
                } catch (...) {
                    errors[i] = std::current_exception();
                }
            });
        }
        for (auto& t : threads) t.join();
    }
    for (auto& e : errors) {
        if (e) std::rethrow_exception(e);
    }

    // Stitch pending-tuple ends to the next worker's first chunk start
    // (or file_size for the trailing worker), preserving file order.
    py::list out;
    for (int i = 0; i < actual_n; ++i) {
        for (auto& c : results[i].chunks) {
            out.append(py::make_tuple(c[0], c[1], c[2], c[3]));
        }
        if (results[i].has_pending) {
            int64_t pend_end = (i + 1 < actual_n) ? starts[i + 1] : file_size;
            out.append(py::make_tuple(results[i].pending_start, pend_end,
                                      results[i].pending_n_crys, results[i].pending_n_obs));
        }
    }
    return out;
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

// ============================================================================
// Native CrystFEL stream writer.
//
// Mirrors write_stream_reconstruct in torchsx/io/_writeback.py byte-for-byte.
// Numeric formatting uses C99 printf, which on glibc applies the same
// round-half-to-even rule as CPython's float __format__, so round-tripping
// (write → re-parse) recovers identical float32 bit patterns.
//
// Throughput is snprintf-bound at ~35 MB/s single-threaded. To scale beyond
// that we use a parallel-format / sequential-drain pipeline:
//
//   for batch of B images:
//     parallel: each of N threads formats B/N images into its per-thread
//               std::string (no I/O, no shared state)
//     serial:   main thread fwrites the per-thread buffers in image order
//
// Memory peak ≈ B × avg_image_bytes (a few MB) regardless of total output
// size — safe for arbitrarily large streams.
// ============================================================================

// Thin sink that appends formatted bytes to a std::string. Mirrors
// FileWriter's interface so format_chunk can be templated over the sink type.
struct StringSink {
    std::string* out;

    explicit StringSink(std::string& s) : out(&s) {}

    void write(const char* data, size_t n) { out->append(data, n); }
    void write(std::string_view s) { out->append(s.data(), s.size()); }
    void write(const std::string& s) { out->append(s); }
    void write(const char* s) { out->append(s); }

    void writef(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
        char buf[512];
        va_list ap;
        va_start(ap, fmt);
        int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        if (n < 0) throw std::runtime_error("vsnprintf failed");
        if (static_cast<size_t>(n) < sizeof(buf)) {
            out->append(buf, static_cast<size_t>(n));
            return;
        }
        std::vector<char> big(static_cast<size_t>(n) + 1);
        va_start(ap, fmt);
        std::vsnprintf(big.data(), big.size(), fmt, ap);
        va_end(ap);
        out->append(big.data(), static_cast<size_t>(n));
    }
};

// Bundle of all the data format_chunk needs. Pointers; no ownership.
struct WriteCtx {
    // per-observation
    const int16_t* hkl;
    const float*   I;
    const float*   sigma;
    const float*   peak;
    const float*   background;
    const float*   fs;
    const float*   ss;
    const int16_t* panel_id;
    // per-crystal
    const float*   cell;
    const float*   astar;
    const float*   bstar;
    const float*   cstar;
    const float*   profile_rad;
    const float*   resolution;
    const int32_t* n_refl;
    const int32_t* image_id;
    // per-image
    const float*   beam_energy;
    const int32_t* n_peaks;
    const float*   det_shift_x;
    const float*   det_shift_y;
    const float*   camera_length;  // may be null
    int64_t        n_cam;
    // strings
    const std::vector<std::string>* filenames;
    const std::vector<std::string>* events;
    const std::vector<std::string>* panel_names;
    const std::string* lattice_type;
    const std::string* centering;
    const std::string* unique_axis;
    // structure
    const std::vector<int64_t>*              obs_off;
    const std::vector<std::vector<int32_t>>* image_to_crys;
    int64_t n_panels;
};

// Format one chunk (Image filename header through "----- End chunk -----")
// by appending bytes to `s`. Templated so the same code emits to either a
// FILE-backed sink (single-threaded direct write) or a string-backed sink
// (multi-threaded format-then-drain).
template <typename Sink>
void format_chunk(Sink& s, int64_t iid, const WriteCtx& c) {
    s.write("----- Begin chunk -----\n");

    if (iid < static_cast<int64_t>(c.filenames->size())) {
        s.writef("Image filename: %s\n", (*c.filenames)[iid].c_str());
    } else {
        s.writef("Image filename: image_%lld\n", static_cast<long long>(iid));
    }
    if (iid < static_cast<int64_t>(c.events->size())) {
        s.writef("Event: %s\n", (*c.events)[iid].c_str());
    } else {
        s.writef("Event: //%lld\n", static_cast<long long>(iid));
    }
    s.writef("Image serial number: %lld\n", static_cast<long long>(iid));
    s.write("hit = 1\n");
    const auto& crys_list = (*c.image_to_crys)[iid];
    s.writef("indexed_by = %s\n", crys_list.empty() ? "none" : "xgandalf");
    s.write("n_indexing_tries = 1\n");
    s.writef("photon_energy_eV = %.6f\n", static_cast<double>(c.beam_energy[iid]));
    s.write("beam_divergence = 0.00e+00 rad\n");
    s.write("beam_bandwidth = 1.00e-08 (fraction)\n");
    double clen_m = (c.camera_length && iid < c.n_cam)
        ? static_cast<double>(c.camera_length[iid]) : 0.0;
    s.writef("average_camera_length = %.6f m\n", clen_m);
    s.writef("num_peaks = %d\n", static_cast<int>(c.n_peaks[iid]));

    s.write("Peaks from peak search\n");
    s.write("  fs/px   ss/px (1/d)/nm^-1   Intensity  Panel\n");
    s.write("End of peak list\n");

    for (int32_t cid : crys_list) {
        s.write("--- Begin crystal\n");

        const float* cp = c.cell + 6 * cid;
        // Cell: a/b/c stored in Å, output in nm (÷10). α/β/γ unchanged.
        // Critical: divide in float32 (matching numpy's weak-scalar promotion
        // for `np.float32_arr / 10.0`), THEN promote to double for snprintf.
        // Doing the divide in double would lose the float32 rounding step
        // that the Python writer applies and produce 1-ULP diffs at .5f.
        double a  = static_cast<double>(cp[0] / 10.0f);
        double b  = static_cast<double>(cp[1] / 10.0f);
        double cc = static_cast<double>(cp[2] / 10.0f);
        double al = static_cast<double>(cp[3]);
        double be = static_cast<double>(cp[4]);
        double ga = static_cast<double>(cp[5]);
        s.writef("Cell parameters %.5f %.5f %.5f nm, %.5f %.5f %.5f deg\n",
                 a, b, cc, al, be, ga);

        const float* ap   = c.astar + 3 * cid;
        const float* bp   = c.bstar + 3 * cid;
        const float* cstp = c.cstar + 3 * cid;
        s.writef("astar = %+.7f %+.7f %+.7f nm^-1\n",
                 static_cast<double>(ap[0] * 10.0f),
                 static_cast<double>(ap[1] * 10.0f),
                 static_cast<double>(ap[2] * 10.0f));
        s.writef("bstar = %+.7f %+.7f %+.7f nm^-1\n",
                 static_cast<double>(bp[0] * 10.0f),
                 static_cast<double>(bp[1] * 10.0f),
                 static_cast<double>(bp[2] * 10.0f));
        s.writef("cstar = %+.7f %+.7f %+.7f nm^-1\n",
                 static_cast<double>(cstp[0] * 10.0f),
                 static_cast<double>(cstp[1] * 10.0f),
                 static_cast<double>(cstp[2] * 10.0f));

        if (!c.lattice_type->empty()) s.writef("lattice_type = %s\n", c.lattice_type->c_str());
        if (!c.centering->empty())    s.writef("centering = %s\n", c.centering->c_str());
        if (!c.unique_axis->empty())  s.writef("unique_axis = %s\n", c.unique_axis->c_str());

        s.writef("profile_radius = %.5f nm^-1\n",
                 static_cast<double>(c.profile_rad[cid] * 10.0f));
        s.write("predict_refine/initial_residual = 0.000000 (0.000000 0.000000 0.000000)\n");
        s.write("predict_refine/final_residual = 0.000000 (0.000000 0.000000 0.000000)\n");
        s.write("predict_refine/total_shifts = 0.000000e+00 0.000000e+00 0.000000e+00\n");
        s.writef("predict_refine/det_shift x = %.3f y = %.3f mm\n",
                 static_cast<double>(c.det_shift_x[iid]),
                 static_cast<double>(c.det_shift_y[iid]));

        double res_A  = static_cast<double>(c.resolution[cid]);
        double res_nm = res_A > 0.0 ? 10.0 / res_A : 0.0;
        s.writef("diffraction_resolution_limit = %.2f nm^-1 or %.2f A\n", res_nm, res_A);

        s.writef("num_reflections = %d\n", static_cast<int>(c.n_refl[cid]));
        s.write("num_saturated_reflections = 0\n");
        s.write("num_implausible_reflections = 0\n");

        s.write("Reflections measured after indexing\n");
        s.write("   h    k    l          I   sigma(I)       peak background  fs/px  ss/px panel\n");

        int64_t lo = (*c.obs_off)[cid];
        int64_t hi = (*c.obs_off)[cid + 1];
        for (int64_t j = lo; j < hi; ++j) {
            int16_t pid = c.panel_id[j];
            const char* pname = (pid >= 0 && pid < c.n_panels)
                ? (*c.panel_names)[pid].c_str()
                : "";
            // Matches Python f-string concat exactly. NO space between bg's
            // 10-wide field and fs's 7-wide field.
            s.writef("%4d %4d %4d %10.2f %10.2f %10.2f %10.2f%7.1f %6.1f %s\n",
                     static_cast<int>(c.hkl[3 * j + 0]),
                     static_cast<int>(c.hkl[3 * j + 1]),
                     static_cast<int>(c.hkl[3 * j + 2]),
                     static_cast<double>(c.I[j]),
                     static_cast<double>(c.sigma[j]),
                     static_cast<double>(c.peak[j]),
                     static_cast<double>(c.background[j]),
                     static_cast<double>(c.fs[j]),
                     static_cast<double>(c.ss[j]),
                     pname);
        }

        s.write("End of reflections\n");
        s.write("--- End crystal\n");
    }

    s.write("----- End chunk -----\n");
}

void write_stream_reconstruct(
    const std::string& path,
    // per-observation arrays — caller must sort by crystal_id
    torch::Tensor hkl,           // (N_obs, 3) int16
    torch::Tensor I,             // (N_obs,)   float32
    torch::Tensor sigma,         // (N_obs,)   float32
    torch::Tensor peak,          // (N_obs,)   float32
    torch::Tensor background,    // (N_obs,)   float32
    torch::Tensor fs,            // (N_obs,)   float32
    torch::Tensor ss,            // (N_obs,)   float32
    torch::Tensor panel_id,      // (N_obs,)   int16
    // per-crystal arrays
    torch::Tensor cell,          // (N_crys, 6) float32, in Å + degrees
    torch::Tensor astar,         // (N_crys, 3) float32, in Å^-1 (×10 in C++)
    torch::Tensor bstar,         // (N_crys, 3) float32, in Å^-1 (×10 in C++)
    torch::Tensor cstar,         // (N_crys, 3) float32, in Å^-1 (×10 in C++)
    torch::Tensor profile_rad,   // (N_crys,)   float32, in Å^-1 (×10 in C++)
    torch::Tensor resolution,    // (N_crys,)   float32, in Å
    torch::Tensor n_refl,        // (N_crys,)   int32 — implies obs slicing
    torch::Tensor image_id,      // (N_crys,)   int32
    // per-image arrays
    torch::Tensor beam_energy,   // (N_img,)    float32, in eV
    torch::Tensor n_peaks,       // (N_img,)    int32
    torch::Tensor det_shift_x,   // (N_img,)    float32, in mm
    torch::Tensor det_shift_y,   // (N_img,)    float32, in mm
    torch::Tensor camera_length, // (N_img,) or empty, float32, in m
    // per-image string metadata (length N_img each)
    const std::vector<std::string>& filenames,
    const std::vector<std::string>& events,
    // panel name lookup (length n_panels), pre-padded with "m{i}" fallbacks
    const std::vector<std::string>& panel_names,
    // optional unit-cell section fields — empty string = absent
    const std::string& lattice_type,
    const std::string& centering,
    const std::string& unique_axis,
    // pre-built header bytes (magic line(s) + geometry + unit cell sections)
    const std::string& header_blob,
    int n_threads,
    int images_per_batch
) {
    const int64_t n_crystals  = n_refl.size(0);
    const int64_t n_images    = beam_energy.size(0);
    const int64_t n_panels    = static_cast<int64_t>(panel_names.size());
    const int64_t n_cam       = camera_length.numel();

    WriteCtx ctx;
    ctx.hkl          = hkl.data_ptr<int16_t>();
    ctx.I            = I.data_ptr<float>();
    ctx.sigma        = sigma.data_ptr<float>();
    ctx.peak         = peak.data_ptr<float>();
    ctx.background   = background.data_ptr<float>();
    ctx.fs           = fs.data_ptr<float>();
    ctx.ss           = ss.data_ptr<float>();
    ctx.panel_id     = panel_id.data_ptr<int16_t>();
    ctx.cell         = cell.data_ptr<float>();
    ctx.astar        = astar.data_ptr<float>();
    ctx.bstar        = bstar.data_ptr<float>();
    ctx.cstar        = cstar.data_ptr<float>();
    ctx.profile_rad  = profile_rad.data_ptr<float>();
    ctx.resolution   = resolution.data_ptr<float>();
    ctx.n_refl       = n_refl.data_ptr<int32_t>();
    ctx.image_id     = image_id.data_ptr<int32_t>();
    ctx.beam_energy  = beam_energy.data_ptr<float>();
    ctx.n_peaks      = n_peaks.data_ptr<int32_t>();
    ctx.det_shift_x  = det_shift_x.data_ptr<float>();
    ctx.det_shift_y  = det_shift_y.data_ptr<float>();
    ctx.camera_length = (n_cam > 0) ? camera_length.data_ptr<float>() : nullptr;
    ctx.n_cam        = n_cam;
    ctx.filenames    = &filenames;
    ctx.events       = &events;
    ctx.panel_names  = &panel_names;
    ctx.lattice_type = &lattice_type;
    ctx.centering    = &centering;
    ctx.unique_axis  = &unique_axis;
    ctx.n_panels     = n_panels;

    // Cumulative observation offsets per crystal.
    std::vector<int64_t> obs_off(static_cast<size_t>(n_crystals) + 1, 0);
    for (int64_t c = 0; c < n_crystals; ++c) {
        obs_off[c + 1] = obs_off[c] + ctx.n_refl[c];
    }
    ctx.obs_off = &obs_off;

    // image_id → ordered list of crystal_ids in that image.
    std::vector<std::vector<int32_t>> image_to_crys(static_cast<size_t>(n_images));
    for (int32_t c = 0; c < n_crystals; ++c) {
        int32_t iid = ctx.image_id[c];
        if (iid >= 0 && iid < n_images) {
            image_to_crys[iid].push_back(c);
        }
    }
    ctx.image_to_crys = &image_to_crys;

    if (n_threads < 1) n_threads = 1;
    if (images_per_batch < 1) images_per_batch = 1024;

    py::gil_scoped_release nogil;

    // Open output with a 1 MiB userspace fwrite buffer.
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        throw std::runtime_error("fopen failed: " + path + ": " + std::strerror(errno));
    }
    std::vector<char> userbuf(1 << 20);
    std::setvbuf(f, userbuf.data(), _IOFBF, userbuf.size());

    auto fwrite_all = [&](const char* data, size_t n) {
        if (n && std::fwrite(data, 1, n, f) != n) {
            std::fclose(f);
            throw std::runtime_error("fwrite failed");
        }
    };

    try {
        fwrite_all(header_blob.data(), header_blob.size());

        // Reusable per-thread format buffers — capacity grows naturally to
        // the largest batch size, then is retained across batches.
        std::vector<std::string> bufs(static_cast<size_t>(n_threads));
        for (auto& b : bufs) b.reserve(1 << 20);

        for (int64_t batch_start = 0; batch_start < n_images; batch_start += images_per_batch) {
            const int64_t batch_end = std::min(
                batch_start + static_cast<int64_t>(images_per_batch), n_images);
            const int64_t batch_n = batch_end - batch_start;

            // Per-thread image ranges within the batch (image-aligned split).
            std::vector<int64_t> tstarts(static_cast<size_t>(n_threads) + 1);
            for (int t = 0; t <= n_threads; ++t) {
                tstarts[t] = batch_start + (batch_n * t) / n_threads;
            }

            if (n_threads == 1) {
                bufs[0].clear();
                StringSink s(bufs[0]);
                for (int64_t iid = batch_start; iid < batch_end; ++iid) {
                    format_chunk(s, iid, ctx);
                }
                fwrite_all(bufs[0].data(), bufs[0].size());
                continue;
            }

            std::vector<std::thread> threads;
            std::vector<std::exception_ptr> errors(static_cast<size_t>(n_threads), nullptr);
            threads.reserve(static_cast<size_t>(n_threads));
            for (int t = 0; t < n_threads; ++t) {
                threads.emplace_back([&, t]() {
                    try {
                        bufs[t].clear();
                        StringSink s(bufs[t]);
                        for (int64_t iid = tstarts[t]; iid < tstarts[t + 1]; ++iid) {
                            format_chunk(s, iid, ctx);
                        }
                    } catch (...) {
                        errors[t] = std::current_exception();
                    }
                });
            }
            for (auto& th : threads) th.join();
            for (auto& e : errors) {
                if (e) { std::fclose(f); std::rethrow_exception(e); }
            }

            // Drain in image order — bufs[0] precedes bufs[1] precedes …
            for (int t = 0; t < n_threads; ++t) {
                fwrite_all(bufs[t].data(), bufs[t].size());
            }
        }
    } catch (...) {
        std::fclose(f);
        throw;
    }
    std::fclose(f);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("count_byte_range", &count_byte_range,
          "Count (n_obs, n_crystals, n_images) in a byte range (mmap).");
    m.def("find_chunk_at_or_after", &find_chunk_at_or_after,
          "Find byte offset of first '----- Begin chunk -----' at or after target.");
    m.def("index_chunks", &index_chunks,
          py::arg("path"), py::arg("n_threads") = 1, py::arg("max_crystals") = -1,
          "Index per-chunk byte offsets and counts. Returns list of "
          "(start, end, n_crys, n_obs) int64 4-tuples in file order. "
          "n_threads>1 splits the file into chunk-aligned byte ranges. "
          "max_crystals>0 enables early-exit head mode (forces single-threaded).");
    m.def("parse_into_shared", &parse_into_shared,
          "Parse a byte range, writing into pre-allocated shared tensors at the given offsets.");
    m.def("write_stream_reconstruct", &write_stream_reconstruct,
          py::arg("path"),
          py::arg("hkl"), py::arg("I"), py::arg("sigma"), py::arg("peak"), py::arg("background"),
          py::arg("fs"), py::arg("ss"), py::arg("panel_id"),
          py::arg("cell"), py::arg("astar"), py::arg("bstar"), py::arg("cstar"),
          py::arg("profile_rad"), py::arg("resolution"), py::arg("n_refl"), py::arg("image_id"),
          py::arg("beam_energy"), py::arg("n_peaks"),
          py::arg("det_shift_x"), py::arg("det_shift_y"), py::arg("camera_length"),
          py::arg("filenames"), py::arg("events"), py::arg("panel_names"),
          py::arg("lattice_type"), py::arg("centering"), py::arg("unique_axis"),
          py::arg("header_blob"),
          py::arg("n_threads") = 1,
          py::arg("images_per_batch") = 1024,
          "Write a CrystFEL .stream file from per-observation/crystal/image tensors. "
          "Caller must sort observations by crystal_id; header_blob is written verbatim "
          "as the file preamble (magic line + geometry + unit-cell sections). "
          "n_threads>1 enables parallel-format / sequential-drain pipeline; memory "
          "peak ≈ images_per_batch × avg_image_bytes, independent of total output size.");
}
