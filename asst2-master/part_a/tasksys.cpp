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
    TaskSystemParallelSpawn::task_num = 0;
    std::thread t[thread_num];


    // Each thread executes task_per_thread tasks
    int tasks_per_thread = num_total_tasks/thread_num;

    auto thread_func = [&](int thread_id, int tasks_per_thread, int num_total_tasks){
        for(;;){
            std::unique_lock<std::mutex> lock(mutex);
            int id = task_num ++;
            lock.unlock();
            
            // If id is valid run, else return
            if (id < num_total_tasks){
                runnable->runTask(id, num_total_tasks);
                continue;
            } else return;
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
            std::unique_lock<std::mutex> lock(mutex);
            int id = task_num ++;

            // skip if increment is greater than available tasks.
            if (id >= TaskSystemParallelThreadPoolSpinning::num_total_tasks) continue; 
            lock.unlock();
            // Execute task
            TaskSystemParallelThreadPoolSpinning::runnable->runTask(id, num_total_tasks);

            // Increment finished tasks counter
            ++TaskSystemParallelThreadPoolSpinning::tasks_finished;
            
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

    
    std::unique_lock<std::mutex> lock(mutex);
    TaskSystemParallelThreadPoolSpinning::runnable = runnable;
    TaskSystemParallelThreadPoolSpinning::num_total_tasks = num_total_tasks;
    TaskSystemParallelThreadPoolSpinning::tasks_finished = 0;
    TaskSystemParallelThreadPoolSpinning::task_num = 0;
    lock.unlock();


    while(tasks_finished<num_total_tasks) std::this_thread::yield();
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
            std::unique_lock<std::mutex> lock(mutex);
            int id = TaskSystemParallelThreadPoolSleeping::task_num++;

            // if id is greater than available tasks, go to bed.
            if (id >= TaskSystemParallelThreadPoolSleeping::num_total_tasks){

                // if id is invalid, sleep
                start.wait(lock);
                continue;
            } 
            lock.unlock();

            // Execute task
            TaskSystemParallelThreadPoolSleeping::runnable->runTask(id, num_total_tasks);

            // increment tasks finished and check for completion before notifying.
            std::unique_lock<std::mutex> lk(mutex2);
            if (++ TaskSystemParallelThreadPoolSleeping::tasks_finished == num_total_tasks){
                finished_flag = true;
                TaskSystemParallelThreadPoolSleeping::cv.notify_all();
            }
            lk.unlock();
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


    std::unique_lock<std::mutex> lock(mutex);
    TaskSystemParallelThreadPoolSleeping::runnable = runnable;
    TaskSystemParallelThreadPoolSleeping::num_total_tasks = num_total_tasks;
    TaskSystemParallelThreadPoolSleeping::tasks_finished = 0;
    TaskSystemParallelThreadPoolSleeping::task_num = 0;
    lock.unlock();
    // notify threads to start
    // printf("!!Waking up now!!\n");
    TaskSystemParallelThreadPoolSleeping::start.notify_all();
    

    // wait to continue
    std::unique_lock<std::mutex> lk(mutex2);
    while(finished_flag == false){
        cv.wait(lk);
    }
    finished_flag = false;
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
