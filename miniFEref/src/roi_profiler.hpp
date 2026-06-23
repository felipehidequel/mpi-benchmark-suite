#ifndef _ROI_PROFILER_HPP_
#define _ROI_PROFILER_HPP_

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <map>
#include <string>
#include <iostream>
#include <iomanip>

namespace miniFE {

class ROIProfiler {
public:
    static ROIProfiler& getInstance() {
        static ROIProfiler instance;
        return instance;
    }

    void start(const std::string& name) {
#ifdef HAVE_MPI
        start_times[name] = MPI_Wtime();
#endif
    }

    void end(const std::string& name) {
#ifdef HAVE_MPI
        double current_time = MPI_Wtime();
        if (start_times.find(name) != start_times.end()) {
            accumulated_times[name] += (current_time - start_times[name]);
        }
#endif
    }

    void print_stats(int myproc, int numprocs) {
#ifdef HAVE_MPI
        if (myproc == 0) {
            std::cout << "\n==========================================================\n";
            std::cout << "               CUSTOM ROI PROFILING REPORT                \n";
            std::cout << "==========================================================\n";
            std::cout << std::left << std::setw(30) << "Region Name" 
                      << std::setw(15) << "Max Time (s)" 
                      << std::setw(15) << "Avg Time (s)" << "\n";
            std::cout << "----------------------------------------------------------\n";
        }

        // Iterate over keys since some procs might have slightly different maps, 
        // but we assume SPMD where all ranks visit the same ROIs.
        for (std::map<std::string, double>::iterator it = accumulated_times.begin(); it != accumulated_times.end(); ++it) {
            const std::string& name = it->first;
            double local_time = it->second;
            double max_time = 0.0;
            double sum_time = 0.0;
            
            MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_time, &sum_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            
            if (myproc == 0) {
                double avg_time = sum_time / numprocs;
                std::cout << std::left << std::setw(30) << name 
                          << std::setw(15) << std::fixed << std::setprecision(6) << max_time 
                          << std::setw(15) << std::fixed << std::setprecision(6) << avg_time << "\n";
            }
        }
        
        if (myproc == 0) {
            std::cout << "==========================================================\n";
        }
#endif
    }

private:
    ROIProfiler() {}
    std::map<std::string, double> start_times;
    std::map<std::string, double> accumulated_times;
};

} // namespace miniFE

#define ROI_START(name) miniFE::ROIProfiler::getInstance().start(name)
#define ROI_END(name)   miniFE::ROIProfiler::getInstance().end(name)

#endif // _ROI_PROFILER_HPP_
