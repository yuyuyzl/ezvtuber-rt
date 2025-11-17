#include <immintrin.h>
#include <cstdint>
#include <omp.h>
#include <stdio.h>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Define ssize_t for MSVC if not available
#ifdef _MSC_VER
    #include <BaseTsd.h>
    typedef SSIZE_T ssize_t;
#endif

template<typename T>
py::array_t<T> create_python_managed_array(const std::vector<ssize_t>& size) {
    ssize_t total_size = 1;
    for (const auto& dim : size) {
        total_size *= dim;
    }
    T* data = new T[total_size];
    auto capsule = py::capsule(data, [](void* ptr) {
        delete[] static_cast<T*>(ptr);
    });
    auto arr = py::array_t<T>(
        size,
        data,
        capsule
    );
    return std::move(arr);
}

/**
 * Split RGBA image into separate RGB and Alpha channels using SSE2 and OpenMP
 * @param rgba_image Input RGBA image buffer (4 bytes per pixel)
 * @param rgb_image Output RGB image buffer (3 bytes per pixel)
 * @param alpha_image Output Alpha channel buffer (1 byte per pixel)
 * @param width Image width in pixels
 * @param height Image height in pixels
 */
void RGBA_Split_SSE2(const uint8_t* rgba_image, uint8_t* rgb_image, uint8_t* alpha_image, int width, int height) {
    const int total_pixels = width * height;
    const int pixels_per_vector = 8;
    const int vectorized_pixels = (total_pixels / pixels_per_vector) * pixels_per_vector;
    
    // Shuffle masks for splitting RGBA -> RGB (packed) and A
    const __m128i rgb_mask = _mm_setr_epi8(
        0, 1, 2,     // R0 G0 B0
        4, 5, 6,     // R1 G1 B1
        8, 9, 10,    // R2 G2 B2
        12, 13, 14,  // R3 G3 B3
        -1, -1, -1, -1
    );
    
    const __m128i alpha_mask = _mm_setr_epi8(
        3, 7, 11, 15,  // A0 A1 A2 A3
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    );
    
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < vectorized_pixels; i += pixels_per_vector) {
            const uint8_t* src_rgba = rgba_image + i * 4;
            uint8_t* dst_rgb = rgb_image + i * 3;
            uint8_t* dst_alpha = alpha_image + i;
            
            // Process 8 pixels as two batches of 4 pixels each
            for (int j = 0; j < 8; j += 4) {
                // Load 4 RGBA pixels (16 bytes)
                __m128i rgba_chunk = _mm_loadu_si128((__m128i*)(src_rgba + j * 4));
                
                // Extract RGB (packed)
                __m128i rgb_packed = _mm_shuffle_epi8(rgba_chunk, rgb_mask);
                
                // Extract Alpha
                __m128i alpha_packed = _mm_shuffle_epi8(rgba_chunk, alpha_mask);
                
                // Store RGB (12 bytes for 4 pixels) - use masked store or manual store to avoid overwrite
                // Store as 8 bytes + 4 bytes to avoid overwriting
                _mm_storel_epi64((__m128i*)(dst_rgb + j * 3), rgb_packed);
                *((uint32_t*)(dst_rgb + j * 3 + 8)) = _mm_extract_epi32(rgb_packed, 2);
                
                // Store Alpha (4 bytes for 4 pixels)
                *((uint32_t*)(dst_alpha + j)) = _mm_cvtsi128_si32(alpha_packed);
            }
        }
        
        // Handle remaining pixels that don't fit in a full vector
        #pragma omp single
        {
            for (int i = vectorized_pixels; i < total_pixels; i++) {
                rgb_image[i * 3 + 0] = rgba_image[i * 4 + 0]; // R
                rgb_image[i * 3 + 1] = rgba_image[i * 4 + 1]; // G
                rgb_image[i * 3 + 2] = rgba_image[i * 4 + 2]; // B
                alpha_image[i] = rgba_image[i * 4 + 3];        // A
            }
        }
    }
}

void RGBA_Combine_SSE2(const uint8_t* rgb_image, const uint8_t* alpha_image, uint8_t* rgba_image, int width, int height) {
    const int total_pixels = width * height;
    const int pixels_per_vector = 8;
    const int vectorized_pixels = (total_pixels / pixels_per_vector) * pixels_per_vector;
    const __m128i rgb_shuffle = _mm_setr_epi8(
        0, 1, 2, -1,
        3, 4, 5, -1,
        6, 7, 8, -1,
        9, 10, 11, -1
    );
    const __m128i alpha_shuffle = _mm_setr_epi8(
        -1, -1, -1, 0,
        -1, -1, -1, 1,
        -1, -1, -1, 2,
        -1, -1, -1, 3
    );
    const __m128i load_mask = _mm_setr_epi32(-1, -1, -1, 0);

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < vectorized_pixels; i += pixels_per_vector) {
            uint8_t* dst_rgba = rgba_image + i * 4;
            const uint8_t* src_rgb = rgb_image + i * 3;
            const uint8_t* src_alpha = alpha_image + i;
            for (int j = 0; j < pixels_per_vector; j += 4) {
                const uint8_t* rgb_ptr = src_rgb + j * 3;
                const uint8_t* alpha_ptr = src_alpha + j;

                __m128i rgb_chunk = _mm_maskload_epi32(reinterpret_cast<const int*>(rgb_ptr), load_mask);
                rgb_chunk = _mm_shuffle_epi8(rgb_chunk, rgb_shuffle);

                uint32_t alpha_scalar = *reinterpret_cast<const uint32_t*>(alpha_ptr);
                __m128i alpha_vals = _mm_cvtsi32_si128(static_cast<int>(alpha_scalar));
                alpha_vals = _mm_shuffle_epi8(alpha_vals, alpha_shuffle);

                __m128i rgba = _mm_or_si128(rgb_chunk, alpha_vals);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_rgba + j * 4), rgba);
            }
        }

        #pragma omp single
        {
            for (int i = vectorized_pixels; i < total_pixels; ++i) {
                rgba_image[i * 4 + 0] = rgb_image[i * 3 + 0];
                rgba_image[i * 4 + 1] = rgb_image[i * 3 + 1];
                rgba_image[i * 4 + 2] = rgb_image[i * 3 + 2];
                rgba_image[i * 4 + 3] = alpha_image[i];
            }
        }
    }
}

/**
 * Convert RGBA to BGRA by swapping R and B channels using SSE2 and OpenMP
 * @param rgba_image Input RGBA image buffer (4 bytes per pixel)
 * @param bgra_image Output BGRA image buffer (4 bytes per pixel)
 * @param width Image width in pixels
 * @param height Image height in pixels
 */
void RGBA_to_BGRA_SSE2(const uint8_t* rgba_image, uint8_t* bgra_image, int width, int height) {
    const int total_pixels = width * height;
    const int pixels_per_vector = 4;
    const int vectorized_pixels = (total_pixels / pixels_per_vector) * pixels_per_vector;
    
    // Shuffle mask to swap R and B channels: RGBA -> BGRA
    // For each pixel: [R, G, B, A] -> [B, G, R, A]
    const __m128i swap_mask = _mm_setr_epi8(
        2, 1, 0, 3,     // B0 G0 R0 A0
        6, 5, 4, 7,     // B1 G1 R1 A1
        10, 9, 8, 11,   // B2 G2 R2 A2
        14, 13, 12, 15  // B3 G3 R3 A3
    );
    
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < vectorized_pixels; i += pixels_per_vector) {
            const uint8_t* src = rgba_image + i * 4;
            uint8_t* dst = bgra_image + i * 4;
            
            // Load 4 RGBA pixels (16 bytes)
            __m128i rgba_chunk = _mm_loadu_si128((__m128i*)src);
            
            // Swap R and B channels
            __m128i bgra_chunk = _mm_shuffle_epi8(rgba_chunk, swap_mask);
            
            // Store 4 BGRA pixels (16 bytes)
            _mm_storeu_si128((__m128i*)dst, bgra_chunk);
        }
        
        // Handle remaining pixels that don't fit in a full vector
        #pragma omp single
        {
            for (int i = vectorized_pixels; i < total_pixels; i++) {
                bgra_image[i * 4 + 0] = rgba_image[i * 4 + 2]; // B
                bgra_image[i * 4 + 1] = rgba_image[i * 4 + 1]; // G
                bgra_image[i * 4 + 2] = rgba_image[i * 4 + 0]; // R
                bgra_image[i * 4 + 3] = rgba_image[i * 4 + 3]; // A
            }
        }
    }
}

/**
 * Convert BGRA to RGBA by swapping B and R channels using SSE2 and OpenMP
 * @param bgra_image Input BGRA image buffer (4 bytes per pixel)
 * @param rgba_image Output RGBA image buffer (4 bytes per pixel)
 * @param width Image width in pixels
 * @param height Image height in pixels
 */
void BGRA_to_RGBA_SSE2(const uint8_t* bgra_image, uint8_t* rgba_image, int width, int height) {
    const int total_pixels = width * height;
    const int pixels_per_vector = 4;
    const int vectorized_pixels = (total_pixels / pixels_per_vector) * pixels_per_vector;
    
    // Shuffle mask to swap B and R channels: BGRA -> RGBA
    // For each pixel: [B, G, R, A] -> [R, G, B, A]
    const __m128i swap_mask = _mm_setr_epi8(
        2, 1, 0, 3,     // R0 G0 B0 A0
        6, 5, 4, 7,     // R1 G1 B1 A1
        10, 9, 8, 11,   // R2 G2 B2 A2
        14, 13, 12, 15  // R3 G3 R3 A3
    );
    
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < vectorized_pixels; i += pixels_per_vector) {
            const uint8_t* src = bgra_image + i * 4;
            uint8_t* dst = rgba_image + i * 4;
            
            // Load 4 BGRA pixels (16 bytes)
            __m128i bgra_chunk = _mm_loadu_si128((__m128i*)src);
            
            // Swap B and R channels
            __m128i rgba_chunk = _mm_shuffle_epi8(bgra_chunk, swap_mask);
            
            // Store 4 RGBA pixels (16 bytes)
            _mm_storeu_si128((__m128i*)dst, rgba_chunk);
        }
        
        // Handle remaining pixels that don't fit in a full vector
        #pragma omp single
        {
            for (int i = vectorized_pixels; i < total_pixels; i++) {
                rgba_image[i * 4 + 0] = bgra_image[i * 4 + 2]; // R
                rgba_image[i * 4 + 1] = bgra_image[i * 4 + 1]; // G
                rgba_image[i * 4 + 2] = bgra_image[i * 4 + 0]; // B
                rgba_image[i * 4 + 3] = bgra_image[i * 4 + 3]; // A
            }
        }
    }
}

/**
 * Convert RGB to BGR by swapping R and B channels using SSE2 and OpenMP
 * Processes 16 pixels (48 bytes) at a time using three XMM registers
 * @param rgb_image Input RGB image buffer (3 bytes per pixel)
 * @param bgr_image Output BGR image buffer (3 bytes per pixel)
 * @param width Image width in pixels
 * @param height Image height in pixels
 */
void RGB_to_BGR_SSE2(const uint8_t* rgb_image, uint8_t* bgr_image, int width, int height){
    const int total_pixels = width * height;
    const int pixels_per_vector = 16;  // 16 pixels = 48 bytes
    const int vectorized_pixels = (total_pixels / pixels_per_vector) * pixels_per_vector;
    
    // Masks for swapping R and B in each 16-byte chunk
    // Let's trace through pixel by pixel:
    // Chunk 1 bytes [0-15]:  R0 G0 B0 | R1 G1 B1 | R2 G2 B2 | R3 G3 B3 | R4 G4 B4 | R5
    //                        0  1  2  | 3  4  5  | 6  7  8  | 9  10 11 | 12 13 14 | 15
    // We want (BGR):         B0 G0 R0 | B1 G1 R1 | B2 G2 R2 | B3 G3 R3 | B4 G4 R4 | B5
    // So shuffle:            2  1  0  | 5  4  3  | 8  7  6  | 11 10 9  | 14 13 12 | ?
    // Byte 15 is R5, it should become B5, but B5 is in chunk2[1]! We can't access it.
    // So we leave byte 15 as-is for now (it will be overwritten by chunk2's result)
    const __m128i mask1 = _mm_setr_epi8(
        2, 1, 0,        // Pixel 0: B0 G0 R0
        5, 4, 3,        // Pixel 1: B1 G1 R1
        8, 7, 6,        // Pixel 2: B2 G2 R2
        11, 10, 9,      // Pixel 3: B3 G3 R3
        14, 13, 12, 15  // Pixel 4: B4 G4 R4, byte 15 stays as-is temporarily
    );
    
    // Chunk 2 bytes [0-15]:  G5 B5 | R6 G6 B6 | R7 G7 B7 | R8 G8 B8 | R9 G9 B9 | R10 G10
    //                        0  1  | 2  3  4  | 5  6  7  | 8  9  10 | 11 12 13 | 14  15
    // We want (BGR):         G5 B5 | B6 G6 R6 | B7 G7 R7 | B8 G8 R8 | B9 G9 R9 | B10 G10
    // So shuffle:            0  1  | 4  3  2  | 7  6  5  | 10 9  8  | 13 12 11 | 14  15
    const __m128i mask2 = _mm_setr_epi8(
        0, 1,           // Pixel 5: G5 B5 (R5 was in chunk1[15], will be handled separately)
        4, 3, 2,        // Pixel 6: B6 G6 R6
        7, 6, 5,        // Pixel 7: B7 G7 R7
        10, 9, 8,       // Pixel 8: B8 G8 R8
        13, 12, 11,     // Pixel 9: B9 G9 R9
        14, 15          // Pixel 10: R10 G10 (B10 is in chunk3[0])
    );
    
    // Chunk 3 bytes [0-15]:  B10 | R11 G11 B11 | R12 G12 B12 | R13 G13 B13 | R14 G14 B14 | R15 G15 B15
    //                        0   | 1   2   3   | 4   5   6   | 7   8   9   | 10  11  12  | 13  14  15
    // We want (BGR):         B10 | B11 G11 R11 | B12 G12 R12 | B13 G13 R13 | B14 G14 R14 | B15 G15 R15
    // So shuffle:            0   | 3   2   1   | 6   5   4   | 9   8   7   | 12  11  10  | 15  14  13
    const __m128i mask3 = _mm_setr_epi8(
        0,              // Pixel 10: B10
        3, 2, 1,        // Pixel 11: B11 G11 R11
        6, 5, 4,        // Pixel 12: B12 G12 R12
        9, 8, 7,        // Pixel 13: B13 G13 R13
        12, 11, 10,     // Pixel 14: B14 G14 R14
        15, 14, 13      // Pixel 15: B15 G15 R15
    );
    
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < vectorized_pixels; i += pixels_per_vector) {
            const uint8_t* src = rgb_image + i * 3;
            uint8_t* dst = bgr_image + i * 3;
            
            // Load 48 bytes (16 RGB pixels)
            __m128i chunk1 = _mm_loadu_si128((__m128i*)(src));
            __m128i chunk2 = _mm_loadu_si128((__m128i*)(src + 16));
            __m128i chunk3 = _mm_loadu_si128((__m128i*)(src + 32));
            
            // Apply shuffle to swap R and B
            __m128i result1 = _mm_shuffle_epi8(chunk1, mask1);
            __m128i result2 = _mm_shuffle_epi8(chunk2, mask2);
            __m128i result3 = _mm_shuffle_epi8(chunk3, mask3);
            
            // Fix cross-chunk bytes
            // Pixel 5: dst[15]=B5, dst[16]=G5, dst[17]=R5
            //          src[15]=R5 (chunk1[15]), src[16]=G5 (chunk2[0]), src[17]=B5 (chunk2[1])
            // Pixel 10: dst[30]=B10, dst[31]=G10, dst[32]=R10
            //           src[30]=R10 (chunk2[14]), src[31]=G10 (chunk2[15]), src[32]=B10 (chunk3[0])
            
            // Extract B5 from chunk2[1] and insert into result1[15]
            uint8_t b5 = _mm_extract_epi8(chunk2, 1);
            result1 = _mm_insert_epi8(result1, b5, 15);
            
            // Extract R5 from chunk1[15] and insert into result2[1]
            uint8_t r5 = _mm_extract_epi8(chunk1, 15);
            result2 = _mm_insert_epi8(result2, r5, 1);
            
            // Extract B10 from chunk3[0] and insert into result2[14]
            uint8_t b10 = _mm_extract_epi8(chunk3, 0);
            result2 = _mm_insert_epi8(result2, b10, 14);
            
            // Extract R10 from chunk2[14] and insert into result3[0]
            uint8_t r10 = _mm_extract_epi8(chunk2, 14);
            result3 = _mm_insert_epi8(result3, r10, 0);
            
            // Store 48 bytes (16 BGR pixels)
            _mm_storeu_si128((__m128i*)(dst), result1);
            _mm_storeu_si128((__m128i*)(dst + 16), result2);
            _mm_storeu_si128((__m128i*)(dst + 32), result3);
        }
        
        // Handle remaining pixels
        #pragma omp single
        {
            for (int i = vectorized_pixels; i < total_pixels; i++) {
                bgr_image[i * 3 + 0] = rgb_image[i * 3 + 2]; // B
                bgr_image[i * 3 + 1] = rgb_image[i * 3 + 1]; // G
                bgr_image[i * 3 + 2] = rgb_image[i * 3 + 0]; // R
            }
        }
    }
}

/**
 * Convert BGR to RGB by swapping B and R channels using SSE2 and OpenMP
 * Processes 16 pixels (48 bytes) at a time using three XMM registers
 * @param bgr_image Input BGR image buffer (3 bytes per pixel)
 * @param rgb_image Output RGB image buffer (3 bytes per pixel)
 * @param width Image width in pixels
 * @param height Image height in pixels
 */
void BGR_to_RGB_SSE2(const uint8_t* bgr_image, uint8_t* rgb_image, int width, int height){
    const int total_pixels = width * height;
    const int pixels_per_vector = 16;  // 16 pixels = 48 bytes
    const int vectorized_pixels = (total_pixels / pixels_per_vector) * pixels_per_vector;
    
    // Same masks as RGB_to_BGR since swapping B<->R is symmetric
    // Chunk 1 bytes [0-15]:  B0 G0 R0 | B1 G1 R1 | B2 G2 R2 | B3 G3 R3 | B4 G4 R4 | B5
    //                        0  1  2  | 3  4  5  | 6  7  8  | 9  10 11 | 12 13 14 | 15
    // We want (RGB):         R0 G0 B0 | R1 G1 B1 | R2 G2 B2 | R3 G3 R3 | R4 G4 B4 | R5
    // So shuffle:            2  1  0  | 5  4  3  | 8  7  6  | 11 10 9  | 14 13 12 | ?
    const __m128i mask1 = _mm_setr_epi8(
        2, 1, 0,        // Pixel 0: R0 G0 B0
        5, 4, 3,        // Pixel 1: R1 G1 B1
        8, 7, 6,        // Pixel 2: R2 G2 B2
        11, 10, 9,      // Pixel 3: R3 G3 B3
        14, 13, 12, 15  // Pixel 4: R4 G4 B4, byte 15 stays as-is temporarily
    );
    
    // Chunk 2 bytes [0-15]:  G5 R5 | B6 G6 R6 | B7 G7 R7 | B8 G8 R8 | B9 G9 R9 | B10 G10
    //                        0  1  | 2  3  4  | 5  6  7  | 8  9  10 | 11 12 13 | 14  15
    // We want (RGB):         G5 R5 | R6 G6 B6 | R7 G7 B7 | R8 G8 B8 | R9 G9 B9 | R10 G10
    // So shuffle:            0  1  | 4  3  2  | 7  6  5  | 10 9  8  | 13 12 11 | 14  15
    const __m128i mask2 = _mm_setr_epi8(
        0, 1,           // Pixel 5: G5 R5 (B5 was in chunk1[15])
        4, 3, 2,        // Pixel 6: R6 G6 B6
        7, 6, 5,        // Pixel 7: R7 G7 B7
        10, 9, 8,       // Pixel 8: R8 G8 B8
        13, 12, 11,     // Pixel 9: R9 G9 B9
        14, 15          // Pixel 10: B10 G10 (R10 is in chunk3[0])
    );
    
    // Chunk 3 bytes [0-15]:  R10 | B11 G11 R11 | B12 G12 R12 | B13 G13 R13 | B14 G14 R14 | B15 G15 R15
    //                        0   | 1   2   3   | 4   5   6   | 7   8   9   | 10  11  12  | 13  14  15
    // We want (RGB):         R10 | R11 G11 B11 | R12 G12 B12 | R13 G13 B13 | R14 G14 B14 | R15 G15 B15
    // So shuffle:            0   | 3   2   1   | 6   5   4   | 9   8   7   | 12  11  10  | 15  14  13
    const __m128i mask3 = _mm_setr_epi8(
        0,              // Pixel 10: R10
        3, 2, 1,        // Pixel 11: R11 G11 B11
        6, 5, 4,        // Pixel 12: R12 G12 B12
        9, 8, 7,        // Pixel 13: R13 G13 B13
        12, 11, 10,     // Pixel 14: R14 G14 B14
        15, 14, 13      // Pixel 15: R15 G15 B15
    );
    
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < vectorized_pixels; i += pixels_per_vector) {
            const uint8_t* src = bgr_image + i * 3;
            uint8_t* dst = rgb_image + i * 3;
            
            // Load 48 bytes (16 BGR pixels)
            __m128i chunk1 = _mm_loadu_si128((__m128i*)(src));
            __m128i chunk2 = _mm_loadu_si128((__m128i*)(src + 16));
            __m128i chunk3 = _mm_loadu_si128((__m128i*)(src + 32));
            
            // Apply shuffle to swap B and R
            __m128i result1 = _mm_shuffle_epi8(chunk1, mask1);
            __m128i result2 = _mm_shuffle_epi8(chunk2, mask2);
            __m128i result3 = _mm_shuffle_epi8(chunk3, mask3);
            
            // Fix cross-chunk bytes
            // Pixel 5: dst[15]=R5, dst[16]=G5, dst[17]=B5
            //          src[15]=B5 (chunk1[15]), src[16]=G5 (chunk2[0]), src[17]=R5 (chunk2[1])
            // Pixel 10: dst[30]=R10, dst[31]=G10, dst[32]=B10
            //           src[30]=B10 (chunk2[14]), src[31]=G10 (chunk2[15]), src[32]=R10 (chunk3[0])
            
            // Extract R5 from chunk2[1] and insert into result1[15]
            uint8_t r5 = _mm_extract_epi8(chunk2, 1);
            result1 = _mm_insert_epi8(result1, r5, 15);
            
            // Extract B5 from chunk1[15] and insert into result2[1]
            uint8_t b5 = _mm_extract_epi8(chunk1, 15);
            result2 = _mm_insert_epi8(result2, b5, 1);
            
            // Extract R10 from chunk3[0] and insert into result2[14]
            uint8_t r10 = _mm_extract_epi8(chunk3, 0);
            result2 = _mm_insert_epi8(result2, r10, 14);
            
            // Extract B10 from chunk2[14] and insert into result3[0]
            uint8_t b10 = _mm_extract_epi8(chunk2, 14);
            result3 = _mm_insert_epi8(result3, b10, 0);
            
            // Store 48 bytes (16 RGB pixels)
            _mm_storeu_si128((__m128i*)(dst), result1);
            _mm_storeu_si128((__m128i*)(dst + 16), result2);
            _mm_storeu_si128((__m128i*)(dst + 32), result3);
        }
        
        // Handle remaining pixels
        #pragma omp single
        {
            for (int i = vectorized_pixels; i < total_pixels; i++) {
                rgb_image[i * 3 + 0] = bgr_image[i * 3 + 2]; // R
                rgb_image[i * 3 + 1] = bgr_image[i * 3 + 1]; // G
                rgb_image[i * 3 + 2] = bgr_image[i * 3 + 0]; // B
            }
        }
    }
}

// ====================================================================================
// Python bindings using pybind11
// ====================================================================================



// Wrapper for RGBA_Split_SSE2 with numpy interface
py::tuple py_RGBA_Split_SSE2(py::array_t<uint8_t> rgba_image) {
    py::buffer_info rgba_buf = rgba_image.request();
    
    if (rgba_buf.ndim != 3 || rgba_buf.shape[2] != 4) {
        throw std::runtime_error("RGBA image must have shape (height, width, 4)");
    }
    
    int height = rgba_buf.shape[0];
    int width = rgba_buf.shape[1];
    
    // Create output arrays
    auto rgb_image = create_python_managed_array<uint8_t>({height, width, 3});
    auto alpha_image = create_python_managed_array<uint8_t>({height, width, 1});
    
    uint8_t* rgba_ptr = static_cast<uint8_t*>(rgba_buf.ptr);
    uint8_t* rgb_ptr = static_cast<uint8_t*>(rgb_image.request().ptr);
    uint8_t* alpha_ptr = static_cast<uint8_t*>(alpha_image.request().ptr);
    
    RGBA_Split_SSE2(rgba_ptr, rgb_ptr, alpha_ptr, width, height);
    
    return py::make_tuple(rgb_image, alpha_image);
}

// Wrapper for RGBA_Combine_SSE2 with numpy interface
py::array_t<uint8_t> py_RGBA_Combine_SSE2(py::array_t<uint8_t> rgb_image, 
                          py::array_t<uint8_t> alpha_image) {
    py::buffer_info rgb_buf = rgb_image.request();
    py::buffer_info alpha_buf = alpha_image.request();
    
    if (rgb_buf.ndim != 3 || rgb_buf.shape[2] != 3) {
        throw std::runtime_error("RGB image must have shape (height, width, 3)");
    }
    
    int height = rgb_buf.shape[0];
    int width = rgb_buf.shape[1];
    
    if (alpha_buf.size != width * height) {
        throw std::runtime_error("Alpha image size mismatch. Expected " + 
                                 std::to_string(width * height) + " bytes.");
    }
    
    // Create output array
    auto rgba_image = create_python_managed_array<uint8_t>({height, width, 4});
    
    uint8_t* rgba_ptr = static_cast<uint8_t*>(rgba_image.request().ptr);
    uint8_t* rgb_ptr = static_cast<uint8_t*>(rgb_buf.ptr);
    uint8_t* alpha_ptr = static_cast<uint8_t*>(alpha_buf.ptr);
    
    RGBA_Combine_SSE2(rgb_ptr, alpha_ptr, rgba_ptr, width, height);
    
    return rgba_image;
}

// Wrapper for RGBA_to_BGRA_SSE2 with numpy interface
py::array_t<uint8_t> py_RGBA_to_BGRA_SSE2(py::array_t<uint8_t> rgba_image) {
    py::buffer_info rgba_buf = rgba_image.request();
    
    if (rgba_buf.ndim != 3 || rgba_buf.shape[2] != 4) {
        throw std::runtime_error("RGBA image must have shape (height, width, 4)");
    }
    
    int height = rgba_buf.shape[0];
    int width = rgba_buf.shape[1];
    
    // Create output array
    auto bgra_image = create_python_managed_array<uint8_t>({height, width, 4});
    
    uint8_t* rgba_ptr = static_cast<uint8_t*>(rgba_buf.ptr);
    uint8_t* bgra_ptr = static_cast<uint8_t*>(bgra_image.request().ptr);
    
    RGBA_to_BGRA_SSE2(rgba_ptr, bgra_ptr, width, height);
    
    return bgra_image;
}

// Wrapper for BGRA_to_RGBA_SSE2 with numpy interface
py::array_t<uint8_t> py_BGRA_to_RGBA_SSE2(py::array_t<uint8_t> bgra_image) {
    py::buffer_info bgra_buf = bgra_image.request();
    
    if (bgra_buf.ndim != 3 || bgra_buf.shape[2] != 4) {
        throw std::runtime_error("BGRA image must have shape (height, width, 4)");
    }
    
    int height = bgra_buf.shape[0];
    int width = bgra_buf.shape[1];
    
    // Create output array
    auto rgba_image = create_python_managed_array<uint8_t>({height, width, 4});
    
    uint8_t* bgra_ptr = static_cast<uint8_t*>(bgra_buf.ptr);
    uint8_t* rgba_ptr = static_cast<uint8_t*>(rgba_image.request().ptr);
    
    BGRA_to_RGBA_SSE2(bgra_ptr, rgba_ptr, width, height);
    
    return rgba_image;
}

// Wrapper for RGB_to_BGR_SSE2 with numpy interface
py::array_t<uint8_t> py_RGB_to_BGR_SSE2(py::array_t<uint8_t> rgb_image) {
    py::buffer_info rgb_buf = rgb_image.request();
    
    if (rgb_buf.ndim != 3 || rgb_buf.shape[2] != 3) {
        throw std::runtime_error("RGB image must have shape (height, width, 3)");
    }
    
    int height = rgb_buf.shape[0];
    int width = rgb_buf.shape[1];
    
    // Create output array
    auto bgr_image = create_python_managed_array<uint8_t>({height, width, 3});
    
    uint8_t* rgb_ptr = static_cast<uint8_t*>(rgb_buf.ptr);
    uint8_t* bgr_ptr = static_cast<uint8_t*>(bgr_image.request().ptr);
    
    RGB_to_BGR_SSE2(rgb_ptr, bgr_ptr, width, height);
    
    return bgr_image;
}

// Wrapper for BGR_to_RGB_SSE2 with numpy interface
py::array_t<uint8_t> py_BGR_to_RGB_SSE2(py::array_t<uint8_t> bgr_image) {
    py::buffer_info bgr_buf = bgr_image.request();
    
    if (bgr_buf.ndim != 3 || bgr_buf.shape[2] != 3) {
        throw std::runtime_error("BGR image must have shape (height, width, 3)");
    }
    
    int height = bgr_buf.shape[0];
    int width = bgr_buf.shape[1];
    
    // Create output array
    auto rgb_image = create_python_managed_array<uint8_t>({height, width, 3});
    
    uint8_t* bgr_ptr = static_cast<uint8_t*>(bgr_buf.ptr);
    uint8_t* rgb_ptr = static_cast<uint8_t*>(rgb_image.request().ptr);
    
    BGR_to_RGB_SSE2(bgr_ptr, rgb_ptr, width, height);
    
    return rgb_image;
}

PYBIND11_MODULE(rgba_utils, m) {
    m.doc() = "SSE2-accelerated image format conversion functions with OpenMP support";
    
    m.def("rgba_split", &py_RGBA_Split_SSE2, 
          "Split RGBA image into separate RGB and Alpha channels. Returns (rgb_image, alpha_image)",
          py::arg("rgba_image"));
    
    m.def("rgba_combine", &py_RGBA_Combine_SSE2,
          "Combine RGB and Alpha channels into RGBA image. Returns rgba_image",
          py::arg("rgb_image"), py::arg("alpha_image"));
    
    m.def("rgba_to_bgra", &py_RGBA_to_BGRA_SSE2,
          "Convert RGBA to BGRA by swapping R and B channels. Returns bgra_image",
          py::arg("rgba_image"));
    
    m.def("bgra_to_rgba", &py_BGRA_to_RGBA_SSE2,
          "Convert BGRA to RGBA by swapping B and R channels. Returns rgba_image",
          py::arg("bgra_image"));
    
    m.def("rgb_to_bgr", &py_RGB_to_BGR_SSE2,
          "Convert RGB to BGR by swapping R and B channels. Returns bgr_image",
          py::arg("rgb_image"));
    
    m.def("bgr_to_rgb", &py_BGR_to_RGB_SSE2,
          "Convert BGR to RGB by swapping B and R channels. Returns rgb_image",
          py::arg("bgr_image"));
}
