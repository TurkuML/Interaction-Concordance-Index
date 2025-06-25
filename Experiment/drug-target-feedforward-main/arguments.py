# dummy class to allow using . notation
class Args():
    pass
ARGS = Args()
# defaults
ARGS.cv_fold="0"
ARGS.setting="S1"
ARGS.batch_size=[64]
ARGS.epochs=[200]
ARGS.num_layers=[3]
ARGS.neurons_in_layers=[1024, 1024, 512]
ARGS.splits_file="splits_davis_RS_2688385916.csv"
ARGS.save_freq=1
ARGS.dropout_ratio=[0.1]
ARGS.dataset="davis"
ARGS.learning_rate=[0.001]

def parse_arg(args, i, ARGS):
    int_param = [
            "--save_freq"
            ]
    float_param = [
            ]
    list_int_param = [
            "--batch_size",
            "--epochs",
            "--num_layers",
            ]
    list_float_param = [
            "--dropout_ratio",
            "--learning_rate"
            ]
    list_list_int_param = [
            "--neurons_in_layers"
            ]
    string_param = [
            "--setting",
            "--cv_fold",
            "--dataset",
            "--splits_file"
            ]

    arg = args[i].lower()
    if arg in int_param:
        try:
            setattr(ARGS, arg[2:], int(args[i+1]))
            return i+2
        except ValueError:
            print("Must provide integer value for",arg)
            exit()
    elif arg in float_param:
        try:
            setattr(ARGS, arg[2:], float(args[i+1]))
            return i+2
        except ValueError:
            print("Must provide float value for",arg)
            exit()
    elif arg in string_param:
        setattr(ARGS, arg[2:], args[i+1])
        return i+2
    elif arg in list_int_param:
        j = 0
        int_args = []
        while True:
            try:
                int_args.append(int(args[i+j+1]))
                j += 1
            except (IndexError, ValueError):
                break
        if j == 0:
            print("Must provide at least one integer value for",arg)
            exit()
        else:
            setattr(ARGS, arg[2:], int_args)
            return i+j+1
    elif arg in list_float_param:
        j = 0
        float_args = []
        while True:
            try:
                float_args.append(float(args[i+j+1]))
                j += 1
            except (IndexError, ValueError):
                break
        if j == 0:
            print("Must provide at least one float value for",arg)
            exit()
        else:
            setattr(ARGS, arg[2:], float_args)
            return i+j+1

    elif arg in list_list_int_param:
        j = 0
        list_int_args = []
        while True:
            try:
                a = args[i+j+1]
                a = a.split(",")
                list_int_args.append([int(x) for x in a])
                j += 1
            except (IndexError, ValueError):
                break
        if j == 0:
            print("Must provide at least one list of int values for",arg)
            exit()
        else:
            setattr(ARGS, arg[2:], list_int_args)
            return i+j+1
    else:
        print("Invalid argument:",arg)
        exit()

def parse_arguments(arglist):
    # NOTE: not all edge cases are properly checked.
    #+the assumption is the user will pass arguments that make sense
    i = 1
    while i < len(arglist):
        i = parse_arg(arglist, i, ARGS)
    return ARGS
