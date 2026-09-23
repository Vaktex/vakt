#include <vector>

namespace app {

template <typename T>
T first(const std::vector<T>& v) {
    return v[0];
}

class Buffer {
public:
    int at(int i) { return data[i]; }
private:
    int data[8];
};

}  // namespace app
