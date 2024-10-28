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
    TaskID taskId; // Unique task identifier
    IRunnable* runnable; // Runnable pointer
    int num_total_tasks; // number of total tasks in bulk launch
    int num_remaining_tasks;
    int task_index = 0;
    int remaining_deps = 0;
    std::unordered_set<TaskID> dependencies;
    std::unordered_set<TaskID> dependents;
    bool finished = false;
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
        TaskID task_id = 0;
        int num_threads;
        bool keep_running = true;                // thread stop conditional
        bool finished = false;                   // thread sync conditional       
        std::atomic<int> num_total_tasks{0};
        IRunnable* runnable;


        std::unordered_map<TaskID, bulkTask> all_tasks;
        std::unordered_map<TaskID, bulkTask*> wait_queue;
        std::unordered_map<TaskID, bulkTask*> ready_queue;

        // std::vector<TaskID> ready_queue;
        // std::vector<TaskID> dep_queue;
        // std::unordered_set<TaskID> finished;
        
        std::unordered_map<TaskID, std::unordered_set<TaskID>> dependencies;
        std::unordered_map<TaskID, std::unordered_set<TaskID>> dependents;
        std::unordered_map<TaskID, int> runs_dep;
        std::unordered_map<TaskID, int> runs;
        std::unordered_map<TaskID, IRunnable*> runnables;
        std::vector<int> t_exited;
        std::atomic<int> task_num{0};
        std::vector<std::thread> t;              // thread pool
        std::mutex mutex;                        // create lock
        std::mutex mutex2;
        std::mutex mutex3;
        std::mutex mutex4;
        std::condition_variable cv;
        std::condition_variable start;     // create condition variable
        std::atomic<uint> tasks_finished{0};           // threads that still haven't finished.

};      

#endif
