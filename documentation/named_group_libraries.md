# Sharing named group definitions across exploration files

The experiment runner now supports importing reusable named group definitions from YAML "library" files so that new exploration configs can reference shared static or variation groups without redefining them each time.

## Creating a library file

1. Create a YAML file with a single document that lists any `named_static_groups` and/or `named_variation_groups` you want to reuse. For example, `explorations/named_group_library.yaml` contains common norm, positional embedding, and head-size groupings.
2. Each entry must include a `named_group` key, and names must be unique across all libraries you plan to combine.

## Using a library in an exploration config

Add a `named_group_libraries` field near the top of your exploration YAML and point to one or more library files (paths are resolved relative to the config file unless absolute):

```yaml
---
named_group_libraries:
  - "named_group_library.yaml"

parameter_groups:
  - named_group_static: ["qk_norm", "pre_ln"]
    named_group_variations:
      - "embedding_head_variations"
      - "positional_embeddings"
```

All named groups from the library become available immediately for use in `named_group_static`, `named_group_variations`, and `named_group_alternates`. If a name collides with one already defined in the config (or another library), the loader raises an error to avoid ambiguity.

## Validation and error handling

* Library files must contain exactly one YAML mapping document; otherwise loading fails.
* Duplicate `named_group` entries across libraries or between a library and the current config result in a clear error message that points to the offending file.
* Missing library paths produce a `FileNotFoundError`, ensuring mis-typed references are caught early.
