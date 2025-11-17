import numpy as np
from ezvtb_rt import ffmpeg_codec
from tqdm import tqdm
import ezvtb_rt.rgba_utils
import cv2

bgra = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)



huff_encoder = ffmpeg_codec.HuffYUVEncoderBGRA(width=512, height=512)
huff_decoder = ffmpeg_codec.HuffYUVDecoderBGRA(512, 512)

for _ in tqdm(range(10000)):
    bgra_encoded = huff_encoder.encode(bgra)

for _ in tqdm(range(10000)):
    bgra_decoded = huff_decoder.decode(bgra_encoded)
print(f'Decoded alpha shape: {bgra_decoded.shape}, dtype: {bgra_decoded.dtype}')
print(np.abs(bgra.astype(np.int16) - bgra_decoded.astype(np.int16)).mean(), np.abs(bgra.astype(np.int16) - bgra_decoded.astype(np.int16)).max())

bgra = cv2.imread('./test/data/base_1.png', cv2.IMREAD_UNCHANGED)
bgra = cv2.resize(bgra, (1024, 1024), interpolation=cv2.INTER_LINEAR)

huff_encoder = ffmpeg_codec.HuffYUVEncoderBGRA(width=1024, height=1024)
huff_decoder = ffmpeg_codec.HuffYUVDecoderBGRA(1024, 1024)

for _ in tqdm(range(10000)):
    bgra_encoded = huff_encoder.encode(bgra)

for _ in tqdm(range(10000)):
    bgra_decoded = huff_decoder.decode(bgra_encoded)
print(f'Decoded alpha shape: {bgra_decoded.shape}, dtype: {bgra_decoded.dtype}')
print(np.abs(bgra.astype(np.int16) - bgra_decoded.astype(np.int16)).mean(), np.abs(bgra.astype(np.int16) - bgra_decoded.astype(np.int16)).max())

cv2.imwrite('./test/data/reconstructed.png', bgra_decoded)
