#include "tasksys.h"


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
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() {
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
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
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
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
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
            if (!keep_running) break;

            // If ready_queue is empty, go to bed
            std::unique_lock<std::mutex> lock(ready_queue_mutex);
            while(ready_queue.empty()&&keep_running){

                start.wait(lock);

                // Need to break if thread is woken up but queue is still empty
                if (!keep_running) break;
            }

            // Lock before fetching task
            bulkTask* task_ptr = nullptr;

            int id;
            // Iterate over queue to find bulk task
            for(auto& bulk_task : ready_queue){
                task_ptr = bulk_task.second;
                
                if (task_ptr->task_index < (task_ptr->num_total_tasks)){

                    // Store id and increment index
                    id = task_ptr->task_index ++;
                    
                    // unlock lock
                    lock.unlock();

                    // rare goto usage
                    goto execute;
                }
            }

            lock.unlock();
            continue;


            execute:
            // Execute task
            task_ptr->runnable->runTask(id, task_ptr->num_total_tasks);

            // After execution, update queues if this is the last id
            int remaining = task_ptr->num_remaining_tasks.fetch_sub(1);
            if (remaining == 1) {

                // Set bulk task to finished
                {
                    std::lock_guard<std::mutex> lock(task_ptr->mutex);
                    task_ptr->finished = true;
                }

                // Update dependents
                {
                    std::lock_guard<std::mutex> lock(all_tasks_mutex);

                    for (auto& dep_id : task_ptr->dependents){
                        bulkTask* dep_task = all_tasks[dep_id];

                        //Remove dependencies
                        {   
                            std::lock_guard<std::mutex> dep_lock(dep_task->mutex);
                            dep_task->dependencies.erase(task_ptr->taskId);

                            // If no more dependencies push to ready queue
                            if (dep_task->dependencies.empty()){
                                {
                                    std::lock_guard<std::mutex> rq_lock(ready_queue_mutex);
                                    ready_queue.emplace(dep_id, dep_task);
                                }
                                start.notify_all();
                            }
                        }
                    }
                

                    // Update ready_queue and notify sync()
                    {
                        std::lock_guard<std::mutex> lock(ready_queue_mutex);
                        ready_queue.erase(task_ptr->taskId);


                        if(ready_queue.empty()){
                            {
                            std::lock_guard<std::mutex> cv_lock(cv_mutex);
                            finished_flag = true;
                            }
                            cv.notify_all();  
                        }

                    }
                }
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

    std::unique_lock<std::mutex> lock(keep_running_mutex);
    // Stop running
    TaskSystemParallelThreadPoolSleeping::keep_running = false;

    // Wake everyone up so they can exit
    TaskSystemParallelThreadPoolSleeping::start.notify_all();

    lock.unlock();

    //Join everything
    int i=0;
    for (auto& thread : TaskSystemParallelThreadPoolSleeping::t){
        thread.join();
        i++;
    }

    for (auto& bulk_task: all_tasks){
        delete bulk_task.second;
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {
    
    // Empty vector
    std::vector<TaskID> noDeps;
    TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(runnable, num_total_tasks, noDeps);
    TaskSystemParallelThreadPoolSleeping::sync();
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {

    // Dynamically allocate bulk_task
    bulkTask* bulk_task = new bulkTask();
    bulk_task->taskId = task_id;
    bulk_task->runnable = runnable;
    bulk_task->num_total_tasks = num_total_tasks;
    bulk_task->num_remaining_tasks = num_total_tasks;


    // Set flag low
    finished_flag = false;

    for (auto& dep : deps) {

        // Fetch dependencies
        bulkTask* dep_task;
        {
            std::lock_guard<std::mutex> lock(all_tasks_mutex);
            dep_task = all_tasks[dep];
        }
        
        // Update dependencies/dependents
        {
            std::lock_guard<std::mutex> lock(dep_task->mutex);
            if (!dep_task->finished) {
                bulk_task->dependencies.insert(dep);
                dep_task->dependents.insert(task_id);
            }
        }
    }
        
    {
        // All tasks are pushed into the all_task map for easy handling
        std::lock_guard<std::mutex> lock(all_tasks_mutex);
        all_tasks.emplace(task_id, bulk_task);
    }


    if (bulk_task->dependencies.empty()) {
        // No dependencies, add to ready_queue
        std::lock_guard<std::mutex> lock(ready_queue_mutex);
        ready_queue.emplace(task_id, bulk_task);
        start.notify_all();
    }

    return task_id ++;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    std::unique_lock<std::mutex> lock(cv_mutex);
    while(finished_flag == false){
        cv.wait(lock);
    }
    return;
}
