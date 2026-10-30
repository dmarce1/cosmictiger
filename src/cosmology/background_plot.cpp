#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <stdexcept>
#include <string>
#include <system_error>

#include <spawn.h>
#include <sys/wait.h>

#include <cosmictiger/background_plot_script.hpp>
#include <cosmictiger/core/safe_io.hpp>
#include <cosmictiger/cosmology/background.hpp>
#include <cosmictiger/cosmology/background_plot.hpp>
#include <cosmictiger/io/output.hpp>

extern char** environ;

namespace cosmology {

namespace detail {

    void renderBackgroundPlot(std::string const& dataPath, std::string const& imagePath) {
        // Pass filenames as argv entries, never as shell or Python source text.
        char* const arguments[] = {const_cast<char*>(kPlotPython), const_cast<char*>("-c"),
            const_cast<char*>(kBackgroundPlotScript), const_cast<char*>(dataPath.c_str()),
            const_cast<char*>(imagePath.c_str()), nullptr};
        pid_t process;
        int const error = posix_spawn(&process, kPlotPython, nullptr, nullptr, arguments, environ);
        if (error != 0) {
            throw std::system_error(error, std::generic_category(), "starting plot renderer");
        }
        int status;
        pid_t result;
        do {
            result = waitpid(process, &status, 0);
        } while (result == -1 && errno == EINTR);
        if (result == -1) {
            throw std::system_error(errno, std::generic_category(), "waiting for plot renderer");
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            throw std::runtime_error("Python renderer failed (Python and Matplotlib are required)");
        }
    }

}    // namespace detail

void writeBackgroundPlot(double finalConformalTime) {
    auto const background = simulationBackground(finalConformalTime);
    auto const dataPath = outputPath("cosmology.txt");
    auto const imagePath = outputPath("cosmology.png");
    // A failed renderer must not leave a plot of a previous run beside new data.
    std::filesystem::remove(imagePath);
    std::ofstream data;
    data.exceptions(std::ios::badbit | std::ios::failbit);
    data.open(dataPath);
    data.imbue(std::locale::classic());
    data << "# tau a adot\n" << std::scientific << std::setprecision(17);
    constexpr int kIntervals = 1024;
    for (int i = 0; i <= kIntervals; i++) {
        double const tau = finalConformalTime * (double(i) / kIntervals);
        auto const state = background->sample(tau).state;
        data << tau << ' ' << state.scaleFactor << ' ' << state.physicalDerivative << '\n';
    }
    data.close();
    try {
        detail::renderBackgroundPlot(dataPath, imagePath);
        PRINT("  Cosmological background plot: %s\n", imagePath.c_str());
    } catch (std::exception const& error) {
        PRINT("  Warning: Unable to render cosmology.png: %s. Samples saved in %s\n", error.what(),
            dataPath.c_str());
    }
}

}    // namespace cosmology
