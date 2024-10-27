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
            while(ready_queue.empty()){
                std::unique_lock<std::mutex> lock(mutex);
                start.wait(lock);

                // Need to break if thread is woken up but queue is still empty
                if (!keep_running) break;
            }

            // Lock before fetching task
            std::unique_lock<std::mutex> lock(mutex);
            bulkTask* task_ptr = nullptr;

            // Mystery variable that will help us later
            bulkTask** task_ptr_in_queue = nullptr;

            uint id;
            // Iterate over queue to find bulk task
            for(auto& bulk_task : ready_queue){
                if (bulk_task->task_index < (bulk_task->num_total_tasks)){
                    // If bulk task isn't finished, take pointer
                    task_ptr = bulk_task;
                    task_ptr_in_queue = &bulk_task;

                    // Store id and increment index
                    id = task_ptr->task_index ++;
                    
                    // unlock lock
                    lock.unlock();
                    // printf("Thread %d took id %d \n", threadId, id);
                    // Use goto
                    goto execute;
                }
            }
            continue;
            lock.unlock();

            execute:
            // Execute task
            task_ptr->runnable->runTask(id, task_ptr->num_total_tasks);
            

            // After execution, update queues if this is the last id
            lock.lock();
            task_ptr->num_remaining_tasks --;
            if (task_ptr->num_remaining_tasks == 0) {
                // Set bulk task to finished
                task_ptr->finished = true;
                // printf("bulk %d Set task finished ptr: %d \n", task_ptr->taskId,task_ptr->finished);
                // Update dependents
                // printf("Dependent size: %ld \n", task_ptr->dependents.size());
                for (auto& dep_id : task_ptr->dependents){
                    // printf("Bulk %d dependent: %d \n", task_ptr->taskId,dep_id);
                    //Remove dependencies
                    all_tasks[dep_id].dependencies.erase(task_ptr->taskId);
                    // printf("Thread %d removed dependency of %d to %d \n",threadId,dep_id,task_ptr->taskId);

                    // If no more dependencies
                    if (all_tasks[dep_id].dependencies.size() == 0){
                        // push to ready queue

                        // printf("Thread %d pushed something to ready queue \n", threadId);
                        ready_queue.push_back(&all_tasks[dep_id]);

                        // delete from wait queue
                        wait_queue.erase(dep_id);
                    }
                }   
                // also iterate over loop and find potential  independent tasks that haven't been scheduled for whatever reason
                // for (auto& [key, value]: wait_queue){
                //     if (value->dependencies.size()==0){
                //         ready_queue.push_back(value);
                //         wait_queue.erase(key);
                //     } //else printf("bulk id %d dep on = %d \n",key, *value->dependencies.begin());
                // }

                // use mystery variable to delete queue element
                ready_queue.erase(std::remove(ready_queue.begin(), ready_queue.end(), *task_ptr_in_queue), ready_queue.end());
            }
            //If queues are empty, notify
            if(ready_queue.empty() && wait_queue.empty()){
                cv.notify_one();  
            }
            lock.unlock();
        }
    };  
    
    // launch threads during construction using above lambda
    uint i = 0;
    for (auto& thread : TaskSystemParallelThreadPoolSleeping::t){
        thread = std::thread(thread_func,i);
        i++;
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {

    // Stop running
    TaskSystemParallelThreadPoolSleeping::keep_running = false;

    // Wake everyone up so they can exit
    TaskSystemParallelThreadPoolSleeping::start.notify_all();

    //Join everything
    // printf("Trying to join");
    uint i=0;
    for (auto& thread : TaskSystemParallelThreadPoolSleeping::t){
        thread.join();
        i++;
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
                                        
    // Create and populate task structure
    bulkTask bulk_task;
    bulk_task.taskId = task_id;
    bulk_task.runnable = runnable;
    bulk_task.num_total_tasks = num_total_tasks;
    bulk_task.num_remaining_tasks = num_total_tasks;

    // Lock while populating queues
    std::unique_lock<std::mutex> lock(mutex);

    // All tasks are pushed into the all_task map for easy handling
    all_tasks.emplace(task_id, bulk_task);

    // Populate dependencies if they exist
    if (deps.size()){

        // Update dependants
        for (auto& dep : deps){

            // Only register dependencies that are active
            if (all_tasks[dep].finished == false){
                all_tasks[task_id].dependencies.insert(dep);
                all_tasks[dep].dependents.insert(task_id);
            }
        }
        // If there are dependencies, push to wait queue
        // else push to ready queue.
        if (all_tasks[task_id].dependencies.size())
            wait_queue.emplace(task_id, &all_tasks[task_id]);
        else
            ready_queue.push_back(&all_tasks[task_id]);
    } else {
        // If no dependencies just add to ready queue
        ready_queue.push_back(&all_tasks[task_id]);
    }
    
    // Unlock 
    lock.unlock();

    // printf("Pushed task id %d  with num_tasks %d into ready queue %ld , or wait queue %ld \n", task_id,num_total_tasks, ready_queue.size(), wait_queue.size());
    // printf("Task 0 statistics: dependencies %ld, dependents %ld\n", all_tasks[0].dependencies.size(), all_tasks[0].dependents.size());
    // printf("Task 1 statistics: dependencies %ld, dependents %ld\n", all_tasks[1].dependencies.size(), all_tasks[1].dependents.size());

    // Notify cause stuff could be sleeping
    start.notify_all();

    return task_id ++;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    std::unique_lock<std::mutex> lock(mutex);
    while(!ready_queue.empty() || !wait_queue.empty()){
        // printf("Called sync, going tobed, ready_queue status = %d, wait_queue status = %d\n", ready_queue.empty(), wait_queue.empty());
        cv.wait(lock);
    }

    return;
}
