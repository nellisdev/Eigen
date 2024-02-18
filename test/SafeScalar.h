
// A Scalar that asserts for uninitialized access.
template <typename T>
class SafeScalar {
 public:
  SafeScalar() : initialized_(false) {}
  SafeScalar(const SafeScalar& other) { *this = other; }
  SafeScalar& operator=(const SafeScalar& other) {
    val_ = T(other);
    initialized_ = true;
    return *this;
  }

  SafeScalar(T val) : val_(val), initialized_(true) {}
  SafeScalar& operator=(T val) {
    val_ = val;
    initialized_ = true;
  }

  operator T() const {
    VERIFY(initialized_ && "Uninitialized access.");
    return val_;
  }

 private:
  T val_;
  bool initialized_;
};

namespace Eigen {
template <typename T>
struct NumTraits<SafeScalar<T>> : GenericNumTraits<T> {
  enum { RequireInitialization = 1 };
};
namespace internal {
template <typename T>
struct make_unsigned<SafeScalar<T>> {
 private:
  using UnsignedT = typename make_unsigned<T>::type;

 public:
  using type = SafeScalar<UnsignedT>;
};
}  // namespace internal
}  // namespace Eigen