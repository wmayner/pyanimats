// asvoid.h

#ifndef ANIMAT_ASVOID_H_
#define ANIMAT_ASVOID_H_

#include <vector>

template <class T>
inline void *asvoid(std::vector<T> *buf)
{
    std::vector<T>& tmp = *buf;
    return (void*)(&tmp[0]);
}

#endif  // ANIMAT_ASVOID_H_
