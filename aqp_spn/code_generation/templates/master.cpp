#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
using namespace std;
namespace py = pybind11;

{methods}

PYBIND11_MODULE(optimized_inference, m){{
    m.doc() = "Generated RSPN ensemble code";
    {registration}
}}