#ifndef _TASKSYS_H
#define _TASKSYS_H

#include "itasksys.h"
#include <thread>
#include <functional>
#include <queue>
#include <mutex>
#include <assert.h>
#include <condition_variable>
#include <atomic>

/*
 * TaskSystemSerial: This class is the student's implementation of a
 * serial task execution engine.  See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemSerial: public ITaskSystem {
    public:
        TaskSystemSerial(int num_threads);
        ~TaskSystemSerial();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();


};

/*
 * TaskSystemParallelSpawn: This class is the student's implementation of a
 * parallel task execution engine that spawns threads in every run()
 * call.  See definition of ITaskSystem in itasksys.h for documentation
 * of the ITaskSystem interface.
 */
class TaskSystemParallelSpawn: public ITaskSystem {
    public:
        TaskSystemParallelSpawn(int num_threads);
        ~TaskSystemParallelSpawn();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();

    private:
        int num_threads;
        std::mutex mutex;
        std::atomic<int> task_num{0};
};

/*
 * TaskSystemParallelThreadPoolSpinning: This class is the student's
 * implementation of a parallel task execution engine that uses a
 * thread pool. See definition of ITaskSystem in itasksys.h for
 * documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSpinning: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSpinning(int num_threads);
        ~TaskSystemParallelThreadPoolSpinning();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();

    private:
        int num_threads;
        bool keep_running = true;                // thread stop conditional
        bool finished = false;
        std::atomic<int> num_total_tasks{0};
        IRunnable* runnable;
        std::atomic<int> task_num{0};
        std::vector<std::thread> t;              // thread pool
        std::deque<std::function<void()>> deque; // queue 
        std::mutex mutex;                        // create lock
        std::mutex mutex2;
        std::mutex mutex3;
        std::condition_variable cv;
        std::condition_variable queue_final;     // create condition variable
        // std::condition_variable finished;        // check if finished.
        std::atomic<int> tasks_finished{0};           // threads that still haven't finished.


};


/*
 * TaskSystemParallelThreadPoolSleeping: This class is the student's
 * optimized implementation of a parallel task execution engine that uses
 * a thread pool. See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSleeping: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSleeping(int num_threads);
        ~TaskSystemParallelThreadPoolSleeping();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();

    private:
        int num_threads;
        bool keep_running = true;                // thread stop conditional
        bool queue_ready = false;
        std::atomic<int> num_total_tasks{0};
        IRunnable* runnable;
        std::atomic<int> task_num{0};
        std::vector<std::thread> t;              // thread pool
        std::deque<std::function<void()>> deque; // queue 
        std::mutex mutex;                        // create lock
        std::mutex mutex2;
        std::mutex mutex3;
        std::condition_variable cv;
        std::condition_variable start;     // create condition variable
        std::condition_variable finished;        // check if finished.
        std::atomic<int> tasks_finished{0};           // threads that still haven't finished.

};

#endif
