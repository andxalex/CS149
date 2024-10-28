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
            std::unique_lock<std::mutex> lock(mutex);
            while(ready_queue.empty()&&keep_running){

                // printf("Thread  %d going to bed \n", threadId);
                start.wait(lock);

                // Need to break if thread is woken up but queue is still empty
                if (!keep_running) break;
            }

            // Lock before fetching task
            bulkTask* task_ptr = nullptr;

            uint id;
            // Iterate over queue to find bulk task
            // printf("Ready queue size: %d \n", ready_queue.size());
            for(auto& bulk_task : ready_queue){
                task_ptr = bulk_task.second;
                // if (taskId == 1)
                // printf("Looking at  bulk id %d = %d, internal index = %d of %d, ready queue size = %d\n",taskId,task_ptr->taskId,task_ptr->task_index,task_ptr->num_total_tasks, ready_queue.size() );
                if (task_ptr->task_index < (task_ptr->num_total_tasks)){
                    // Store id and increment index
                    id = task_ptr->task_index ++;
                    
                    // unlock lock
                    lock.unlock();
                    // printf("Thread %d took id %d \n", threadId, id);
                    // Use goto
                    goto execute;
                }
            }
            // printf("Spinning?? \n");
            continue;
            lock.unlock();

            execute:
            // Execute task
            task_ptr->runnable->runTask(id, task_ptr->num_total_tasks);
            // printf("Executed task bulk %d id %d of %d\n",task_ptr->taskId, id,task_ptr->num_total_tasks);

            // After execution, update queues if this is the last id
            lock.lock();
            task_ptr->num_remaining_tasks --;
            if (task_ptr->num_remaining_tasks == 0) {
                // printf("Entered here\n");
                // Set bulk task to finished
                task_ptr->finished = true;
                // printf("bulk %d Set task finished ptr: %d \n", task_ptr->taskId,task_ptr->finished);
                // Update dependents
                // printf("Dependent size: %ld \n", task_ptr->dependents.size());
                for (auto& dep_id : task_ptr->dependents){
                    // printf("Bulk %d dependent: %d \n", task_ptr->taskId,dep_id);
                    //Remove dependencies
                    all_tasks[dep_id]->dependencies.erase(task_ptr->taskId);
                    // printf("Thread %d removed dependency of %d to %d \n",threadId,dep_id,task_ptr->taskId);

                    // If no more dependencies
                    if (all_tasks[dep_id]->dependencies.size() == 0){
                        // push to ready queue

                        // printf("Thread %d pushed something to ready queue \n", threadId);
                        // ready_queue.emplace(dep_id, &all_tasks[dep_id]);
                        ready_queue.emplace(dep_id, all_tasks[dep_id]);
                        // printf("Pushed block task id %d on ready queue, this has an id of %d\n",dep_id, all_tasks[dep_id].taskId);

                    }
                }   

                // use mystery variable to delete queue element
                // printf("BEFORE: Deleting bulk task id %d, ready_queue size = %d \n", task_ptr->taskId, ready_queue.size());
                ready_queue.erase(task_ptr->taskId);
                // printf("AFTER: Deleting bulk task id %d, ready_queue size = %d \n", task_ptr->taskId, ready_queue.size());

                // printf("Here, task id %d, %d of %d, ready_queue = %d empty? %d, wait_queue = %d empty? %d \n",task_ptr->taskId, id, task_ptr->num_total_tasks, ready_queue.size(), ready_queue.empty(), wait_queue.size(), wait_queue.empty());
                if(ready_queue.empty()){
                    finished_flag = true;
                    // printf("Notified");
                    cv.notify_all();  
                }
            }
            //If queues are empty, notify
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

    std::unique_lock<std::mutex> lock(mutex);
    // Stop running
    TaskSystemParallelThreadPoolSleeping::keep_running = false;

    // Wake everyone up so they can exit
    TaskSystemParallelThreadPoolSleeping::start.notify_all();

    lock.unlock();
    //Join everything
    // printf("Trying to join");
    uint i=0;
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

    // Lock while populating queues
    std::unique_lock<std::mutex> lock(mutex);

    // Set flag low
    finished_flag = false;

    // Populate dependencies if they exist
    // if (ready_queue.size()== 0){

    // }
    if (deps.size()){

        // Update dependants
        for (auto& dep : deps){

            // Only register dependencies that are active
            if (all_tasks[dep]->finished == false){
                bulk_task->remaining_deps ++;
                bulk_task->dependencies.insert(dep);
                all_tasks[dep]->dependents.insert(task_id);
            }
        }
    }
        

    // All tasks are pushed into the all_task map for easy handling
    all_tasks.emplace(task_id, bulk_task);

    // If no dependencies, then push to ready queue
    if (bulk_task->remaining_deps == 0){
        ready_queue.emplace(task_id, bulk_task);
    }

    // Unlock 
    lock.unlock();

    // Notify cause stuff could be sleeping
    start.notify_all();

    return task_id ++;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    std::unique_lock<std::mutex> lock(mutex);

    while(finished_flag == false){
        // printf("Called sync, going tobed, ready_queue status = %d, wait_queue status = %d\n", ready_queue.empty(), wait_queue.empty());
        cv.wait(lock);
    }
    // printf("synced \n");
    return;
}
