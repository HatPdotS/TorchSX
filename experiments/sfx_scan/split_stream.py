from torchsx.stream import SFXStream



stream = '/das/work/p17/p17490/Peter/torchsx/sfx_scan/partialator/neg.stream'

s = SFXStream.from_stream(stream, spacegroup='P1', cache='/das/work/p17/p17490/Peter/torchsx/sfx_scan/partialator/neg.sfx', n_workers=4, merge_friedel=False)



s1, s2 = s.split_random(n_parts=2, seed=42)

s1.write_stream('/das/work/p17/p17490/Peter/torchsx/sfx_scan/partialator/neg_half1.stream')
s2.write_stream('/das/work/p17/p17490/Peter/torchsx/sfx_scan/partialator/neg_half2.stream')