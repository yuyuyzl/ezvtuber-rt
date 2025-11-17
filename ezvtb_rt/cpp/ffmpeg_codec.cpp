#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cstring>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>
#include <libswscale/swscale.h>
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace {
constexpr uint32_t HUFFYUV_MAGIC = 0x48554646; // 'HUFF'
}

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

// Initialize FFmpeg logging - call once
static bool ffmpeg_log_initialized = false;
static void init_ffmpeg_logging() {
    if (!ffmpeg_log_initialized) {
        // Set log level: AV_LOG_QUIET (suppress all), AV_LOG_ERROR (errors only), AV_LOG_WARNING, AV_LOG_INFO
        av_log_set_level(AV_LOG_ERROR);  // Change to AV_LOG_QUIET to suppress all messages
        ffmpeg_log_initialized = true;
    }
}


// HuffYUV Encoder BGRA Class - Fast truly lossless encoder for BGRA images
class HuffYUVEncoderBGRA {
private:
    int width_;
    int height_;
    AVCodecContext* codec_ctx_;
    AVFrame* frame_;
    AVPacket* pkt_;
    std::vector<uint8_t> buffer_;
    std::vector<uint8_t> extradata_;
    bool initialized_;

public:
    HuffYUVEncoderBGRA(int width, int height) 
        : width_(width), height_(height),
          codec_ctx_(nullptr), frame_(nullptr), 
          pkt_(nullptr), initialized_(false) {
        
        init_ffmpeg_logging();
        buffer_.resize(16 * 1024 * 1024); // 16 MB buffer
        initialize();
    }

    ~HuffYUVEncoderBGRA() {
        cleanup();
    }

    void initialize() {
        if (initialized_) {
            return;
        }

        // Use HuffYUV codec for truly lossless BGRA encoding
        const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_HUFFYUV);
        if (!codec) {
            throw std::runtime_error("HuffYUV codec not found");
        }

        // Create codec context
        codec_ctx_ = avcodec_alloc_context3(codec);
        if (!codec_ctx_) {
            throw std::runtime_error("Could not allocate codec context");
        }

        // Set encoding parameters for lossless encoding
        codec_ctx_->width = width_;
        codec_ctx_->height = height_;
        codec_ctx_->time_base = AVRational{1, 25};
        codec_ctx_->framerate = AVRational{25, 1};
        codec_ctx_->pix_fmt = AV_PIX_FMT_BGRA;  // Native BGRA format - truly lossless

        // HuffYUV is always lossless, no quality settings needed

        // Open codec
        if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
            cleanup();
            throw std::runtime_error("Could not open codec");
        }

        // Allocate frame
        frame_ = av_frame_alloc();
        if (!frame_) {
            cleanup();
            throw std::runtime_error("Could not allocate frame");
        }

        frame_->format = codec_ctx_->pix_fmt;
        frame_->width = width_;
        frame_->height = height_;
        frame_->pts = 0;

        // Allocate buffer for the frame
        if (av_frame_get_buffer(frame_, 0) < 0) {
            cleanup();
            throw std::runtime_error("Could not allocate frame buffer");
        }

        // Allocate packet for encoded data
        pkt_ = av_packet_alloc();
        if (!pkt_) {
            cleanup();
            throw std::runtime_error("Could not allocate packet");
        }

        if (codec_ctx_->extradata && codec_ctx_->extradata_size > 0) {
            extradata_.assign(
                codec_ctx_->extradata,
                codec_ctx_->extradata + codec_ctx_->extradata_size
            );
        } else {
            extradata_.clear();
        }

        initialized_ = true;
    }

    void cleanup() {
        if (pkt_) {
            av_packet_free(&pkt_);
            pkt_ = nullptr;
        }
        if (frame_) {
            av_frame_free(&frame_);
            frame_ = nullptr;
        }
        if (codec_ctx_) {
            avcodec_free_context(&codec_ctx_);
            codec_ctx_ = nullptr;
        }
        initialized_ = false;
    }

    py::array_t<uint8_t> encode(py::array_t<uint8_t> bgra_image) {
        if (!initialized_) {
            throw std::runtime_error("Encoder not initialized");
        }

        // Request buffer info from numpy array
        py::buffer_info buf_info = bgra_image.request();
        
        // Validate input dimensions
        if (buf_info.ndim != 3) {
            throw std::runtime_error("Image must be a 3D array (height, width, channels)");
        }
        
        int height = static_cast<int>(buf_info.shape[0]);
        int width = static_cast<int>(buf_info.shape[1]);
        int channels = static_cast<int>(buf_info.shape[2]);
        
        if (width != width_ || height != height_) {
            throw std::runtime_error("Image size mismatch. Expected " + 
                std::to_string(width_) + "x" + std::to_string(height_) + 
                " but got " + std::to_string(width) + "x" + std::to_string(height));
        }
        
        if (channels != 4) {
            throw std::runtime_error("Channel count mismatch. Expected 4 channels (BGRA) but got " + 
                std::to_string(channels));
        }

        // Get pointer to image data
        const uint8_t* bgra_data = static_cast<uint8_t*>(buf_info.ptr);

        // Copy BGRA data directly to frame (no color conversion needed!)
        for (int y = 0; y < height_; y++) {
            std::memcpy(frame_->data[0] + y * frame_->linesize[0],
                       bgra_data + y * width_ * 4,
                       width_ * 4);
        }

        // Increment PTS for each frame
        frame_->pts++;
        
        // Encode the frame
        int ret = avcodec_send_frame(codec_ctx_, frame_);
        if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
            throw std::runtime_error(std::string("Error sending frame for encoding: ") + errbuf);
        }

        // Retrieve encoded packets
        ssize_t total_written = 0;
        bool got_packet = false;
        while (true) {
            ret = avcodec_receive_packet(codec_ctx_, pkt_);
            if (ret == AVERROR(EAGAIN)) {
                break;
            } else if (ret == AVERROR_EOF) {
                break;
            } else if (ret < 0) {
                char errbuf[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
                throw std::runtime_error(std::string("Error receiving encoded packet: ") + errbuf);
            }

            got_packet = true;
            
            // Check if buffer has enough space
            if (total_written + pkt_->size > buffer_.size()) {
                av_packet_unref(pkt_);
                throw std::runtime_error("Encoding buffer too small");
            }

            // Copy packet data to output buffer
            std::memcpy(buffer_.data() + total_written, pkt_->data, pkt_->size);
            total_written += pkt_->size;
            av_packet_unref(pkt_);
        }

        // If we didn't get any packets, something is wrong
        if (!got_packet && total_written == 0) {
            throw std::runtime_error("Encoder did not produce any output (got AVERROR(EAGAIN)). This may indicate an encoder configuration issue.");
        }

        // Create output numpy array with the encoded data
        const uint32_t extradata_size = static_cast<uint32_t>(extradata_.size());
        const ssize_t header_size = sizeof(uint32_t) * 2; // magic + extradata size
        const ssize_t total_output_size = header_size + extradata_size + total_written;
        py::array_t<uint8_t> result = create_python_managed_array<uint8_t>({total_output_size});
        py::buffer_info result_buf = result.request();
        uint8_t* result_ptr = static_cast<uint8_t*>(result_buf.ptr);
        
        // Write header (magic + extradata size)
        uint32_t magic = HUFFYUV_MAGIC;
        std::memcpy(result_ptr, &magic, sizeof(uint32_t));
        std::memcpy(result_ptr + sizeof(uint32_t), &extradata_size, sizeof(uint32_t));
        ssize_t offset = header_size;

        // Write extradata if any
        if (extradata_size > 0) {
            std::memcpy(result_ptr + offset, extradata_.data(), extradata_size);
            offset += extradata_size;
        }
        
        // Copy encoded data to output array after header
        std::memcpy(result_ptr + offset, buffer_.data(), total_written);
        
        return result;
    }

    int get_width() const { return width_; }
    int get_height() const { return height_; }
};

// HuffYUV Decoder BGRA Class - Fast truly lossless decoder for BGRA images
class HuffYUVDecoderBGRA {
private:
    int width_;
    int height_;
    AVCodecContext* codec_ctx_;
    AVFrame* frame_;
    AVPacket* pkt_;
    std::vector<uint8_t> buffer_;
    std::vector<uint8_t> extradata_;
    bool initialized_;

public:
    HuffYUVDecoderBGRA(int width, int height) 
        : width_(width), height_(height),
          codec_ctx_(nullptr), frame_(nullptr), 
          pkt_(nullptr), initialized_(false) {
        
        init_ffmpeg_logging();
        buffer_.resize(16 * 1024 * 1024); // 16 MB buffer
        // Don't initialize here - do it lazily on first decode
    }

    ~HuffYUVDecoderBGRA() {
        cleanup();
    }

    void initialize() {
        if (initialized_) {
            return;
        }

        // Use HuffYUV decoder
        const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_HUFFYUV);
        if (!codec) {
            throw std::runtime_error("HuffYUV decoder not found");
        }
        
        codec_ctx_ = avcodec_alloc_context3(codec);
        if (!codec_ctx_) {
            throw std::runtime_error("Could not allocate decoder context");
        }
        
        // HuffYUV bitstreams produced by avcodec don't include dimension metadata,
        // so we must provide the expected frame info up front.
        codec_ctx_->width = width_;
        codec_ctx_->height = height_;
        codec_ctx_->pix_fmt = AV_PIX_FMT_BGRA;

        if (!extradata_.empty()) {
            codec_ctx_->extradata = static_cast<uint8_t*>(
                av_mallocz(extradata_.size() + AV_INPUT_BUFFER_PADDING_SIZE));
            if (!codec_ctx_->extradata) {
                cleanup();
                throw std::runtime_error("Could not allocate HuffYUV extradata buffer");
            }
            std::memcpy(codec_ctx_->extradata, extradata_.data(), extradata_.size());
            codec_ctx_->extradata_size = static_cast<int>(extradata_.size());
        }

        int ret = avcodec_open2(codec_ctx_, codec, nullptr);
        if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
            cleanup();
            throw std::runtime_error(std::string("Could not open HuffYUV decoder: ") + errbuf);
        }
        
        // Allocate packet for encoded data
        pkt_ = av_packet_alloc();
        if (!pkt_) {
            cleanup();
            throw std::runtime_error("Could not allocate packet");
        }
        
        // Allocate frame for decoded data
        frame_ = av_frame_alloc();
        if (!frame_) {
            cleanup();
            throw std::runtime_error("Could not allocate frame");
        }

        initialized_ = true;
    }

    void cleanup() {
        if (pkt_) {
            av_packet_free(&pkt_);
            pkt_ = nullptr;
        }
        if (frame_) {
            av_frame_free(&frame_);
            frame_ = nullptr;
        }
        if (codec_ctx_) {
            avcodec_free_context(&codec_ctx_);
            codec_ctx_ = nullptr;
        }
        initialized_ = false;
    }

    py::array_t<uint8_t> decode(py::array_t<uint8_t> encoded_bytes) {
        // Request buffer info from numpy array
        py::buffer_info buf_info = encoded_bytes.request();
        
        // Validate input dimensions
        if (buf_info.ndim != 1) {
            throw std::runtime_error("Input must be a 1D array of encoded bytes");
        }
        
        const uint8_t* encoded_data = static_cast<uint8_t*>(buf_info.ptr);
        ssize_t encoded_size = buf_info.shape[0];

        if (encoded_size < static_cast<ssize_t>(sizeof(uint32_t) * 2)) {
            throw std::runtime_error("Encoded HuffYUV buffer too small (missing header)");
        }

        uint32_t magic = 0;
        uint32_t extradata_size = 0;
        std::memcpy(&magic, encoded_data, sizeof(uint32_t));
        std::memcpy(&extradata_size, encoded_data + sizeof(uint32_t), sizeof(uint32_t));

        if (magic != HUFFYUV_MAGIC) {
            throw std::runtime_error("Invalid HuffYUV buffer magic header");
        }

        ssize_t offset = sizeof(uint32_t) * 2;
        if (encoded_size < offset + static_cast<ssize_t>(extradata_size)) {
            throw std::runtime_error("Encoded HuffYUV buffer missing extradata payload");
        }

        const uint8_t* extradata_ptr = encoded_data + offset;
        offset += extradata_size;
        if (encoded_size <= offset) {
            throw std::runtime_error("Encoded HuffYUV buffer missing encoded payload");
        }

        const uint8_t* payload_ptr = encoded_data + offset;
        ssize_t payload_size = encoded_size - offset;

        // Initialize lazily using parsed extradata
        if (!initialized_) {
            extradata_.assign(extradata_ptr, extradata_ptr + extradata_size);
            initialize();
        }
        
        // Unref any previous packet data
        av_packet_unref(pkt_);
        
        // Set packet data
        pkt_->data = const_cast<uint8_t*>(payload_ptr);
        pkt_->size = static_cast<int>(payload_size);
        
        // Send packet to decoder
        int ret = avcodec_send_packet(codec_ctx_, pkt_);
        if (ret == AVERROR_EOF) {
            // Decoder was in EOF state, flush and reset it
            avcodec_flush_buffers(codec_ctx_);
            // Try sending packet again
            ret = avcodec_send_packet(codec_ctx_, pkt_);
        }
        
        if (ret < 0 && ret != AVERROR(EAGAIN)) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
            throw std::runtime_error(std::string("Error sending packet to decoder: ") + errbuf);
        }
        
        // Receive decoded frame
        ret = avcodec_receive_frame(codec_ctx_, frame_);
        
        if (ret == AVERROR(EAGAIN)) {
            // Need to flush the decoder to get the frame
            avcodec_send_packet(codec_ctx_, nullptr);
            ret = avcodec_receive_frame(codec_ctx_, frame_);
            
            if (ret == AVERROR(EAGAIN)) {
                throw std::runtime_error("No frame decoded - decoder needs more data (AVERROR(EAGAIN))");
            } else if (ret == AVERROR_EOF) {
                // After flushing, reset for next decode
                avcodec_flush_buffers(codec_ctx_);
                throw std::runtime_error("No frame decoded - end of stream after flush (AVERROR_EOF)");
            } else if (ret < 0) {
                char errbuf[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
                throw std::runtime_error(std::string("Error receiving frame after flush: ") + errbuf);
            }
        } else if (ret == AVERROR_EOF) {
            throw std::runtime_error("No frame decoded - end of stream (AVERROR_EOF)");
        } else if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errbuf, AV_ERROR_MAX_STRING_SIZE);
            throw std::runtime_error(std::string("Error receiving frame from decoder: ") + errbuf);
        }
        
        // Get frame dimensions and format
        int width = frame_->width;
        int height = frame_->height;
        AVPixelFormat frame_format = static_cast<AVPixelFormat>(frame_->format);
        
        // Verify dimensions match expected
        if (width != width_ || height != height_) {
            throw std::runtime_error("Decoded frame dimensions mismatch. Expected " + 
                std::to_string(width_) + "x" + std::to_string(height_) + 
                " but got " + std::to_string(width) + "x" + std::to_string(height));
        }
        
        // Verify it's BGRA format
        if (frame_format != AV_PIX_FMT_BGRA) {
            throw std::runtime_error("Decoded frame is not in BGRA format. Got format: " + 
                std::to_string(static_cast<int>(frame_format)));
        }
        
        int channels = 4;  // BGRA
        
        // Check if buffer is large enough
        ssize_t required_size = width * height * channels;
        if (required_size > buffer_.size()) {
            buffer_.resize(required_size);  // Auto-resize if needed
        }
        
        // Copy BGRA data directly (no conversion needed!)
        for (int y = 0; y < height; y++) {
            std::memcpy(buffer_.data() + y * width * channels,
                       frame_->data[0] + y * frame_->linesize[0],
                       width * channels);
        }
        
        // Create output numpy array with appropriate shape
        py::array_t<uint8_t> result = create_python_managed_array<uint8_t>({height, width, channels});
        py::buffer_info result_buf = result.request();
        uint8_t* result_ptr = static_cast<uint8_t*>(result_buf.ptr);
        
        // Copy decoded data to output array
        std::memcpy(result_ptr, buffer_.data(), required_size);
        
        // Unref the frame to prepare for next decode
        av_frame_unref(frame_);
        
        return result;
    }
};


// Pybind11 module definition
PYBIND11_MODULE(ffmpeg_codec, m) {
    m.doc() = "FFmpeg H.264 encoding and decoding module for real-time video processing";
    // HuffYUVEncoderBGRA class binding
    py::class_<HuffYUVEncoderBGRA>(m, "HuffYUVEncoderBGRA")
        .def(py::init<int, int>(),
             py::arg("width"),
             py::arg("height"),
             R"pbdoc(
                 Initialize HuffYUV Encoder for truly lossless encoding of BGRA images.
                 
                 HuffYUV is a fast, truly lossless codec that works directly with BGRA format,
                 avoiding any color space conversion losses. This encoder reuses internal 
                 resources for maximum performance. Only supports 4-channel BGRA images 
                 in (height, width, 4) format.
                 
                 Parameters
                 ----------
                 width : int
                     Width of images to encode
                 height : int
                     Height of images to encode
                 
                 Examples
                 --------
                 >>> import numpy as np
                 >>> from ezvtb_rt import ffmpeg_codec
                 >>> # Create encoder for 512x512 BGRA images
                 >>> encoder = ffmpeg_codec.HuffYUVEncoderBGRA(512, 512)
                 >>> img = np.random.randint(0, 256, (512, 512, 4), dtype=np.uint8)
                 >>> encoded = encoder.encode(img)
             )pbdoc")
        .def("encode", &HuffYUVEncoderBGRA::encode,
             py::arg("bgra_image"),
             R"pbdoc(
                 Truly losslessly encode a BGRA image to HuffYUV format.
                 
                 HuffYUV uses native BGRA format with no color space conversion, 
                 ensuring zero pixel differences after encoding and decoding.
                 
                 Parameters
                 ----------
                 bgra_image : numpy.ndarray
                     Input BGRA image with shape (height, width, 4) matching
                     the dimensions specified during initialization.
                 
                 Returns
                 -------
                 numpy.ndarray
                     Encoded HuffYUV byte sequence as a 1D numpy array of uint8.
                 
                 Raises
                 ------
                 RuntimeError
                     If image dimensions don't match, channels != 4, or encoding fails.
             )pbdoc")
        .def_property_readonly("width", &HuffYUVEncoderBGRA::get_width)
        .def_property_readonly("height", &HuffYUVEncoderBGRA::get_height);
    
    // HuffYUVDecoderBGRA class binding
    py::class_<HuffYUVDecoderBGRA>(m, "HuffYUVDecoderBGRA")
        .def(py::init<int, int>(),
             py::arg("width"),
             py::arg("height"),
             R"pbdoc(
                 Initialize HuffYUV Decoder for decoding HuffYUV encoded images to BGRA.
                 
                 This decoder requires knowing the frame dimensions in advance and always outputs
                 4-channel BGRA images. It reuses internal resources (codec context, frames)
                 for optimal performance. Works with truly lossless HuffYUV BGRA encoding.
                 
                 Parameters
                 ----------
                 width : int
                     Width of images to decode
                 height : int
                     Height of images to decode
                 
                 Examples
                 --------
                 >>> import numpy as np
                 >>> from ezvtb_rt import ffmpeg_codec
                 >>> # Create a decoder for 512x512 images
                 >>> decoder = ffmpeg_codec.HuffYUVDecoderBGRA(512, 512)
                 >>> # Decode HuffYUV encoded data to BGRA
                 >>> decoded = decoder.decode(encoded_bytes)
                 >>> print(f"Decoded shape: {decoded.shape}")  # (512, 512, 4)
             )pbdoc")
        .def("decode", &HuffYUVDecoderBGRA::decode,
             py::arg("encoded_bytes"),
             R"pbdoc(
                 Decode HuffYUV encoded bytes back to a BGRA image.
                 
                 Automatically detects the image size and returns 4-channel BGRA with
                 zero pixel differences from the original (truly lossless).
                 
                 Parameters
                 ----------
                 encoded_bytes : numpy.ndarray
                     Input encoded HuffYUV byte sequence as a 1D numpy array of uint8.
                 
                 Returns
                 -------
                 numpy.ndarray
                     Decoded BGRA image as a numpy array with shape (height, width, 4).
                 
                 Raises
                 ------
                 RuntimeError
                     If decoding fails or input is invalid.
             )pbdoc");
}
