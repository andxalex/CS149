#include "tasksys.h"
#include <iostream>
#include <thread>
#include <stdio.h>

IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemSerial::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //

    // store thread num
    TaskSystemParallelSpawn::num_threads = num_threads;
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    // Define a number of threads
    int thread_num = TaskSystemParallelSpawn::num_threads;
    std::thread t[thread_num];


    // Each thread executes task_per_thread tasks
    int tasks_per_thread = num_total_tasks/thread_num;

    auto thread_func = [&](int thread_id, int tasks_per_thread, int num_total_tasks){
        
        // Compute indices
        int start = thread_id*tasks_per_thread;
        int end = (thread_id == (thread_num - 1))? num_total_tasks : (start+tasks_per_thread);

        // Each thread executes a number of tasks
        for (int i=start; i<end; i++){
            runnable->runTask(i, num_total_tasks);
        }
    };

    // Launch threads
    for (int i = 0; i<thread_num; i++){
        t[i] = std::thread(thread_func, i, tasks_per_thread, num_total_tasks);
    }

    // Join threads
    for (int i=0; i < thread_num; i++){
        t[i].join();
    }

}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).

    // Set thread number
    TaskSystemParallelThreadPoolSpinning::num_threads = num_threads;
    
    // Resize thread array based on thread num
    TaskSystemParallelThreadPoolSpinning::t.resize(num_threads);

    // Init
    TaskSystemParallelThreadPoolSpinning::task_num = 0;
    TaskSystemParallelThreadPoolSpinning::tasks_finished = 0;

    // Define worker lambda
    auto thread_func = [&](int threadId){
        for (;;){

            // Destructor condition
            if (TaskSystemParallelThreadPoolSpinning::keep_running == false)
                break;

            // Get id of task to start working on
            int id = TaskSystemParallelThreadPoolSpinning::task_num.fetch_add(1);

            // skip if increment is greater than available tasks.
            if (id >= TaskSystemParallelThreadPoolSpinning::num_total_tasks) continue; 

            // Execute task
            TaskSystemParallelThreadPoolSpinning::runnable->runTask(id, num_total_tasks);

            // Increment finished tasks counter
            TaskSystemParallelThreadPoolSpinning::tasks_finished.fetch_add(1);
        }
    };  
    
    // launch threads during construction using above lambda
    int i = 0;
    for (auto& thread : TaskSystemParallelThreadPoolSpinning::t){
        thread = std::thread(thread_func,i);
        i++;
    }
}


TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {

    // Stop running
    TaskSystemParallelThreadPoolSpinning::keep_running = false;

    //Join everything
    for (auto& thread : TaskSystemParallelThreadPoolSpinning::t){
        thread.join();
    }
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {

    TaskSystemParallelThreadPoolSpinning::runnable = runnable;
    TaskSystemParallelThreadPoolSpinning::num_total_tasks = num_total_tasks;
    TaskSystemParallelThreadPoolSpinning::tasks_finished = 0;
    TaskSystemParallelThreadPoolSpinning::task_num = 0;



    for(;;){
        int val = TaskSystemParallelThreadPoolSpinning::tasks_finished.load();
        if (val >=num_total_tasks)
            break;
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // You do not need to implement this method.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads) {

    // Set thread number
    TaskSystemParallelThreadPoolSleeping::num_threads = num_threads;
    
    // Resize thread array based on thread num
    TaskSystemParallelThreadPoolSleeping::t.resize(num_threads);

    // Define worker lambda
    auto thread_func = [&](int threadId){
        for (;;){
            // Destructor condition
            if (TaskSystemParallelThreadPoolSleeping::keep_running == false)
                break;

            // Get id of task to start working on
            int id = TaskSystemParallelThreadPoolSleeping::task_num.fetch_add(1);

            // if id is greater than available tasks, go to bed.
            if (id >= TaskSystemParallelThreadPoolSleeping::num_total_tasks){

                // if id is invalid, sleep
                std::unique_lock<std::mutex> lock(TaskSystemParallelThreadPoolSleeping::mutex);
                start.wait(lock);

                continue;
            } 

            // Execute task
            TaskSystemParallelThreadPoolSleeping::runnable->runTask(id, num_total_tasks);

            // increment tasks finished and check for completion before notifying.
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (++ TaskSystemParallelThreadPoolSleeping::tasks_finished == num_total_tasks)
                    TaskSystemParallelThreadPoolSleeping::cv.notify_all();
            }
        }
    };  
    
    // launch threads during construction using above lambda
    int i = 0;
    for (auto& thread : TaskSystemParallelThreadPoolSleeping::t){
        thread = std::thread(thread_func,i);
        i++;
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {

    // Stop running
    TaskSystemParallelThreadPoolSleeping::keep_running = false;

    // Wake up sleeping threads
    TaskSystemParallelThreadPoolSleeping::start.notify_all();

    //Join everything
    for (auto& thread : TaskSystemParallelThreadPoolSleeping::t){
        thread.join();
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    TaskSystemParallelThreadPoolSleeping::runnable = runnable;
    TaskSystemParallelThreadPoolSleeping::num_total_tasks = num_total_tasks;
    TaskSystemParallelThreadPoolSleeping::tasks_finished = 0;
    TaskSystemParallelThreadPoolSleeping::task_num = 0;

    // notify threads to start
    // printf("!!Waking up now!!\n");
    TaskSystemParallelThreadPoolSleeping::start.notify_all();
    

    // wait to continue
    int unum = num_total_tasks;
    {
        std::unique_lock<std::mutex> lk(mutex);
        cv.wait(lk, [&](){ return tasks_finished.load() >= unum; });
    }
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}
