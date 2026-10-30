#pragma once

namespace cosmology {

// Write the full configured history, including the signed physical da/dt.
// Rendering failures leave the text samples available and report a warning.
void writeBackgroundPlot(double finalConformalTime);

}    // namespace cosmology
