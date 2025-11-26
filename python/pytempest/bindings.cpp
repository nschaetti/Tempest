#include <Python.h>

#include <stdexcept>
#include <string>

#include "anfwi/anfwi_runner.hpp"
#include "tempest/init.hpp"

namespace {

using tempest::anfwi::SimulationStats;

void raise_runtime_error(const std::string& msg) {
    PyErr_SetString(PyExc_RuntimeError, msg.c_str());
}

bool set_dict_item(PyObject* dict, const char* key, PyObject* value) {
    if (!value) {
        Py_DECREF(dict);
        return false;
    }
    const int rc = PyDict_SetItemString(dict, key, value);
    Py_DECREF(value);
    if (rc != 0) {
        Py_DECREF(dict);
        return false;
    }
    return true;
}

PyObject* config_to_dict(const SimulationConfig& cfg) {
    PyObject* dict = PyDict_New();
    if (!dict) {
        return nullptr;
    }

    if (!set_dict_item(dict, "nx", PyLong_FromLong(cfg.nx)) ||
        !set_dict_item(dict, "nz", PyLong_FromLong(cfg.nz)) ||
        !set_dict_item(dict, "nt", PyLong_FromLong(cfg.nt)) ||
        !set_dict_item(dict, "dx", PyFloat_FromDouble(cfg.dx)) ||
        !set_dict_item(dict, "dt", PyFloat_FromDouble(cfg.dt)) ||
        !set_dict_item(dict, "c0", PyFloat_FromDouble(cfg.c0)) ||
        !set_dict_item(dict, "block_size_x", PyLong_FromLong(cfg.block_size_x)) ||
        !set_dict_item(dict, "block_size_y", PyLong_FromLong(cfg.block_size_y)) ||
        !set_dict_item(dict, "display_scale", PyFloat_FromDouble(cfg.display_scale)) ||
        !set_dict_item(dict, "display_interval", PyLong_FromLong(cfg.display_interval))) {
        return nullptr;
    }

    return dict;
}

PyObject* stats_to_dict(const SimulationStats& stats) {
    PyObject* dict = PyDict_New();
    if (!dict) {
        return nullptr;
    }

    if (!set_dict_item(dict, "steps", PyLong_FromLong(stats.steps)) ||
        !set_dict_item(dict, "elapsed_seconds", PyFloat_FromDouble(stats.elapsed_seconds)) ||
        !set_dict_item(dict, "final_peak", PyFloat_FromDouble(stats.final_peak))) {
        return nullptr;
    }

    return dict;
}

int read_int_field(PyObject* dict, const char* key, int default_value, bool required) {
    PyObject* value = PyDict_GetItemString(dict, key);
    if (!value) {
        if (required) {
            throw std::runtime_error(std::string("Missing integer field '") + key + "'");
        }
        return default_value;
    }

    long tmp = PyLong_AsLong(value);
    if (tmp == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        throw std::runtime_error(std::string("Field '") + key + "' must be an integer");
    }
    return static_cast<int>(tmp);
}

float read_float_field(PyObject* dict, const char* key, float default_value, bool required) {
    PyObject* value = PyDict_GetItemString(dict, key);
    if (!value) {
        if (required) {
            throw std::runtime_error(std::string("Missing float field '") + key + "'");
        }
        return default_value;
    }

    double tmp = PyFloat_AsDouble(value);
    if (tmp == -1.0 && PyErr_Occurred()) {
        PyErr_Clear();
        throw std::runtime_error(std::string("Field '") + key + "' must be numeric");
    }
    return static_cast<float>(tmp);
}

SimulationConfig config_from_dict(PyObject* dict) {
    if (!PyDict_Check(dict)) {
        throw std::runtime_error("Configuration must be provided as a dict");
    }

    SimulationConfig cfg;
    cfg.nx = read_int_field(dict, "nx", 0, true);
    cfg.nz = read_int_field(dict, "nz", 0, true);
    cfg.nt = read_int_field(dict, "nt", 0, true);
    cfg.dx = read_float_field(dict, "dx", 1.0f, true);
    cfg.dt = read_float_field(dict, "dt", 1.0f, true);
    cfg.c0 = read_float_field(dict, "c0", 1500.0f, false);
    cfg.block_size_x = read_int_field(dict, "block_size_x", 16, false);
    cfg.block_size_y = read_int_field(dict, "block_size_y", 16, false);
    cfg.display_scale = read_float_field(dict, "display_scale", 1.0f, false);
    cfg.display_interval = read_int_field(dict, "display_interval", 10, false);
    return cfg;
}

SimulationConfig config_from_py(PyObject* obj) {
    if (PyUnicode_Check(obj)) {
        const char* path = PyUnicode_AsUTF8(obj);
        if (!path) {
            throw std::runtime_error("Configuration path must be valid UTF-8");
        }
        return tempest::anfwi::load_config(path);
    }
    if (PyDict_Check(obj)) {
        return config_from_dict(obj);
    }
    throw std::runtime_error("Configuration must be a path string or a dict");
}

} // namespace

static PyObject* pytempest_load_config(PyObject*, PyObject* args) {
    const char* path = nullptr;
    if (!PyArg_ParseTuple(args, "s", &path)) {
        return nullptr;
    }

    try {
        const auto cfg = tempest::anfwi::load_config(path);
        return config_to_dict(cfg);
    } catch (const std::exception& ex) {
        raise_runtime_error(ex.what());
        return nullptr;
    }
}

static PyObject* pytempest_run_simulation(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* config_obj = nullptr;
    int display = 0;
    static const char* kwlist[] = {"config", "display", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", const_cast<char**>(kwlist),
                                     &config_obj, &display)) {
        return nullptr;
    }

    try {
        SimulationConfig cfg = config_from_py(config_obj);
        const auto stats = tempest::anfwi::run_anfwi_simulation(cfg, display != 0);
        return stats_to_dict(stats);
    } catch (const std::exception& ex) {
        raise_runtime_error(ex.what());
        return nullptr;
    }
}

static PyMethodDef PyTempestMethods[] =
{
    {"load_config", pytempest_load_config, METH_VARARGS,
     "load_config(path) -> dict\nReturn a SimulationConfig mapping loaded from disk."},
    {"run_simulation", reinterpret_cast<PyCFunction>(pytempest_run_simulation),
     METH_VARARGS | METH_KEYWORDS,
     "run_simulation(config, *, display=False) -> dict\nRun the ANFWI simulation and "
     "return statistics."},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef PyTempestModule {
    PyModuleDef_HEAD_INIT,
    "pytempest",
    "Minimal bindings for the Tempest CPU simulators.",
    -1,
    PyTempestMethods
};

extern "C" PyMODINIT_FUNC PyInit_pytempest(void) {
    return PyModule_Create(&PyTempestModule);
}
