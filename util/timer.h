#ifndef TIMER_H_INCLUDED
#define TIMER_H_INCLUDED
#include <sys/time.h>

struct Timer{
        Timer() {} ;
        void start();
        double stop();
    
        struct timeval now;
        struct timeval then;
};

inline void Timer::start() {
    gettimeofday(&then,NULL);
}

inline double Timer::stop() {
    gettimeofday(&now,NULL);
    return (now.tv_sec - then.tv_sec + 1e-6 * (now.tv_usec - then.tv_usec));
    //return (now.tv_sec - then.tv_sec) * 1e3 + (1e-3)*(now.tv_usec-then.tv_usec);
}


#endif // TIMER_H_INCLUDED
