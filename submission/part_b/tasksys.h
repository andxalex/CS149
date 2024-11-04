#ifndef _TASKSYS_H
#define _TASKSYS_H

#include "itasksys.h"
#include <algorithm>
#include <thread>
#include <functional>
#include <queue>
#include <mutex>
#include <assert.h>
#include <condition_variable>
#include <atomic>
#include <unordered_map>
#include <unordered_set>

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
};


/*
 * Task structure
 */
struct bulkTask {
    TaskID taskId;                          // Unique task identifier
    IRunnable* runnable;                    // Runnable pointer
    int num_total_tasks;                    // number of total tasks in bulk launch
    std::atomic<int> num_remaining_tasks;   // number of remaining tasks in bulk launch
    int task_index = 0;                     // subtask index
    std::unordered_set<TaskID> dependencies;// dependencies
    std::unordered_set<TaskID> dependents;  // dependents
    bool finished = false;                  // bulk finish flag
    std::mutex mutex;                       // bulk mutex
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
        TaskID task_id = 0;                                 // internal task id
        int num_threads;                                    // number of threads
        bool keep_running = true;                           // thread stop conditional
        bool finished_flag = false;                         // thread sync conditional       
        std::unordered_map<TaskID, bulkTask*> all_tasks;    // stores all tasks always
        std::unordered_map<TaskID, bulkTask*> ready_queue;  // execution queue
        std::vector<std::thread> t;                         // thread pool
        std::mutex ready_queue_mutex;                       // mutex for ready_queue map
        std::mutex all_tasks_mutex;                         // mutex for all_gasks map
        std::mutex keep_running_mutex;                      // mutex for destructor
        std::mutex cv_mutex;                                // mutex for cv
        std::condition_variable cv;                         // sync condition variable
        std::condition_variable start;                      // thread wake up cv


};      

#endif
