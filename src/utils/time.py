import time

def get_time_status(start_t, total_step, current_step):
    current_t = time.time()

    elapsed_time = current_t - start_t
    elapsed_days = int(elapsed_time // (3600 * 24))
    elapsed_hours = int((elapsed_time % (3600 * 24)) // 3600)
    elapsed_minutes = int((elapsed_time % 3600) // 60)
    elapsed_seconds = int(elapsed_time % 60)

    if current_step > 0:
        average_step_time = elapsed_time / current_step
    else:
        average_step_time = 0

    if current_step < total_step:
        remaining_steps = total_step - current_step
        remaining_time = remaining_steps * average_step_time

        remaining_days = int(remaining_time // (3600 * 24))
        remaining_hours = int((remaining_time % (3600 * 24)) // 3600)
        remaining_minutes = int((remaining_time % 3600) // 60)
        remaining_seconds = int(remaining_time % 60)
    else:
        remaining_days = 0
        remaining_hours = 0
        remaining_minutes = 0
        remaining_seconds = 0

    elapsed_str = f"{elapsed_days}d {elapsed_hours:02d}h {elapsed_minutes:02d}m {elapsed_seconds:02d}s"
    remaining_str = f"{remaining_days}d {remaining_hours:02d}h {remaining_minutes:02d}m {remaining_seconds:02d}s"
    average_step_str = f"{average_step_time:.2f}s/step"
    
    return elapsed_str, remaining_str, average_step_str