# MultimodalStudio: Modularity Documentation

MultimodalStudio is designed with a highly modular architecture to enable flexible experimentation and easy extension. This guide explains how modules are structured, how to define new methods, and how to customize existing methods using configuration files.

## 1. Module Structure: Config and Implementation

Each module in MultimodalStudio is composed of two main parts:

- **Config Class**: This class defines the configuration for the module. It contains parameters that control the module's behavior. These parameters can be:
  - Primitive types (e.g., int, float, str, bool)
  - Other Config classes (enabling hierarchical composition of modules)

- **Implementation Class**: This is the actual class that implements the module's logic.

Both the Config and Implementation classes are coded in the same file for each module. You can extend either class by creating subclasses, allowing for further customization and specialization.

To instantiate a module, you call the `setup()` method on its Config class. This method returns an instance of the implementation class, initialized with the specified configuration.

**Example:**
```python
my_config = SomeModuleConfig(param1=..., param2=...)
my_module = my_config.setup()
```

This design ensures that modules are decoupled and can be easily swapped or reconfigured.

## 2. Predefined Method Configurations

The file `src/configs/method_configs.py` contains a dictionary called `method_configs` with several predefined method configurations. Each entry in this dictionary specifies:

- The set of modules to use for a particular method (i.e., a pipeline)
- Optionally, the specific parameters for each module

A method configuration is essentially a tree of Config objects, each specifying which module to use and how to configure it.

If a parameter is not explicitly set in a method configuration, the default value specified in the corresponding Config class will be used.

## 3. Defining New Methods

To define a new method (i.e., a new pipeline with a different set of modules), you can add a new entry to the `method_configs` dictionary in `method_configs.py`. In this entry, you specify:

- Which modules to use for each component of the pipeline
- Any specific parameters for those modules

**Example:**
```python
method_configs['my_new_method'] = TrainerConfig(
    method_name='my_new_method',
    pipeline=BasePipelineConfig(
        datamanager=DataManagerConfig(...),
        model=BaseModelConfig(...),
        # ... other components ...
    ),
    # ... other TrainerConfig parameters ...
)
```

This approach allows you to fully customize the architecture by composing different modules.

## 4. Customizing Methods with Configuration Files

If you want to change some parameters of a method **without modifying the set of modules**, you can use a YAML configuration file (see examples in the `./confs` directory).

- The YAML file can overwrite only variables that are numerical, string, or boolean values, or data structures (like dictionaries or lists) containing these types.
- **You cannot swap out a module for a different one using a YAML file.** The set of modules is fixed by the method configuration in `method_configs.py`.
- If you do not explicitly overwrite a parameter in a YAML config, the value from the method configuration will be used. If the method configuration does not specify a value, the default from the Config class will be used.

**Example YAML:**
```yaml
pipeline:
  model:
    surface_model:
      surface_field:
        field:
          hidden_dim: 512
      compute_hessian: false
```

This will override the `hidden_dim` and `compute_hessian` parameters for the specified modules, but will not change which modules are used.

## 5. Best Practices

- Use YAML config files for quick parameter tuning and experiments.
- Use `method_configs.py` to define new architectures or swap modules.
- Keep modules small and focused; use Config classes to compose complex behaviors.
- Extend Config or Implementation classes with subclasses as needed for advanced customization.

For more details, refer to the code in `src/configs/method_configs.py` and the example configuration files in `./confs`.
