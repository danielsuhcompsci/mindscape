Before pushing, if you installed any dependencies do:

```
conda env export > environment.yml
```

To create

```
conda env create -f environment.yml
```

To update yml

```
conda env update --name <env_name> --file environment.yml --prune
```
