#ifndef MMAP_UTIL_H_
#define MMAP_UTIL_H_

#ifdef __unix__

#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace cnpy {

class MMapFile {
public:
  // Open file and map it into memory. mode = "r" for read-only, "rw" for
  // read-write.
  MMapFile(const std::string &path, const std::string &mode = "r")
      : fd_(-1), data_(nullptr), size_(0) {
    readonly_ = (mode == "r");
    int flags = readonly_ ? O_RDONLY : O_RDWR;
    fd_ = ::open(path.c_str(), flags);
    if (fd_ == -1) {
      throw std::runtime_error("MMapFile: cannot open file " + path);
    }
    struct stat st;
    if (fstat(fd_, &st) == -1) {
      ::close(fd_);
      throw std::runtime_error("MMapFile: cannot stat file " + path);
    }
    size_ = static_cast<size_t>(st.st_size);
    int prot = readonly_ ? PROT_READ : PROT_READ | PROT_WRITE;
    void *map = ::mmap(nullptr, size_, prot, MAP_SHARED, fd_, 0);
    if (map == MAP_FAILED) {
      ::close(fd_);
      throw std::runtime_error("MMapFile: mmap failed for file " + path);
    }
    data_ = static_cast<char *>(map);
  }
  // Open and map an already open file descriptor. mode = "r" for read-only, "rw" for
  // read-write.
  MMapFile(int fd, const std::string &mode = "r")
      : fd_(-1), data_(nullptr), size_(0) {
    if (fd < 0 || fd == STDIN_FILENO || fd == STDOUT_FILENO || fd == STDERR_FILENO) {
      throw std::invalid_argument("MMapFile: invalid file descriptor");
    }
    readonly_ = (mode == "r");
    struct stat st;
    if (fstat(fd, &st) == -1) {
      ::close(fd);
      throw std::runtime_error("MMapFile: cannot stat file descriptor " + std::to_string(fd));
    }
    size_ = static_cast<size_t>(st.st_size);
    int prot = readonly_ ? PROT_READ : PROT_READ | PROT_WRITE;
    void *map = ::mmap(nullptr, size_, prot, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
      ::close(fd);
      throw std::runtime_error("MMapFile: mmap failed for file descriptor " + std::to_string(fd));
    }
    data_ = static_cast<char *>(map);
    fd_ = fd;
  }

  // Disable copy
  MMapFile(const MMapFile &) = delete;
  MMapFile &operator=(const MMapFile &) = delete;

  // Enable move
  MMapFile(MMapFile &&other) noexcept
      : fd_(other.fd_), data_(other.data_), size_(other.size_) {
    other.fd_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
  }

  MMapFile &operator=(MMapFile &&other) noexcept {
    if (this != &other) {
      unmap();
      fd_ = other.fd_;
      data_ = other.data_;
      size_ = other.size_;
      other.fd_ = -1;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  ~MMapFile() { unmap(); }

  const char *data() const { return data_; }
  char *data() { return data_; }
  size_t size() const { return size_; }
  bool is_open() const { return data_ != nullptr; }
  bool is_readonly() const { return readonly_; }

private:
  void unmap() {
    if (data_ && size_ > 0) {
      ::munmap(data_, size_);
      data_ = nullptr;
    }
    if (fd_ != -1) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  int fd_;
  char *data_;
  bool readonly_;
  size_t size_;
};

} // namespace cnpy

#endif // __unix__

#endif // MMAP_UTIL_H_