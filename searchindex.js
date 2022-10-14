Search.setIndex({"docnames": ["build", "config", "faq", "index", "installation", "limitations", "logging", "references", "tutorial"], "filenames": ["build.rst", "config.rst", "faq.rst", "index.rst", "installation.rst", "limitations.rst", "logging.rst", "references.rst", "tutorial.rst"], "titles": ["Building from source", "Configurations", "FAQs", "RaNNC (Rapid Neural Network Connector)", "Installation", "Limitations", "Logging", "API references", "Tutorial"], "terms": {"you": [0, 1, 2, 3, 5, 6, 7, 8], "must": [0, 1, 4, 5, 7, 8], "us": [0, 1, 3, 4, 5, 6, 7], "gcc": [0, 4], "v5": 0, "4": [0, 4], "newer": 0, "rannc": [0, 1, 4, 5, 6, 7], "ha": [0, 2, 3, 4, 5, 6, 7, 8], "been": [0, 4], "test": [0, 4, 5, 8], "v8": 0, "2": [0, 1, 4], "note": [0, 2, 7, 8], "built": [0, 4], "complianc": 0, "abi": 0, "pytorch": [0, 2, 3, 4, 5, 7, 8], "i": [0, 1, 3, 5, 7, 8], "pre": 0, "cxx11": 0, "_glibcxx_use_cxx11_abi": 0, "0": [0, 1, 3, 4, 7], "follow": [0, 1, 4, 5, 6, 7, 8], "binari": 0, "packag": [0, 4], "can": [0, 1, 3, 5, 6, 7, 8], "chang": [0, 2, 7], "set": [0, 1, 2, 6, 7], "cmakelist": 0, "txt": 0, "provid": 0, "function": [0, 5], "below": [0, 3, 8], "understand": 0, "how": [0, 8], "clone": 0, "repositori": 0, "submodul": 0, "recurs": 0, "requir": [0, 3, 4, 8], "git": 0, "http": [0, 4], "github": [0, 4], "com": 0, "nict": [0, 4], "wisdom": [0, 4], "need": [0, 2, 3, 8], "some": [0, 3], "environ": [0, 1, 4], "befor": [0, 7, 8], "help": [0, 8], "cmake": 0, "find": [0, 2, 7], "depend": 0, "librari": [0, 4, 7, 8], "cuda_hom": 0, "path": [0, 1, 2, 7], "cuda": [0, 3, 4, 7, 8], "runtim": [0, 1, 4, 8], "directori": 0, "mpi_dir": 0, "an": [0, 1, 2, 3, 5, 6, 7, 8], "mpi": [0, 4, 8], "boost_dir": 0, "boost": 0, "cudnn_root_dir": 0, "cudnn": 0, "ld_library_path": [0, 8], "contain": 0, "nccl": [0, 4, 8], "The": [0, 1, 2, 3, 4, 6, 7, 8], "process": [0, 2, 3, 7, 8], "refer": 0, "conda": 0, "therefor": [0, 8], "your": [0, 2, 4, 5], "python": [0, 2, 4, 8], "run": [0, 1, 2, 3, 7], "setup": 0, "py": [0, 5, 8], "script": [0, 5, 8], "show": [0, 1, 2, 3, 4, 5, 6, 7, 8], "usr": 0, "bin": [0, 1], "env": 0, "bash": 0, "activ": [0, 8], "conda_path": 0, "etc": [0, 8], "profil": [0, 8], "d": [0, 6], "sh": [0, 4], "export": 0, "dirnam": 0, "which": [0, 3], "nvcc": 0, "ompi_info": 0, "boost_dir_path": 0, "your_cudnn_dir_path": 0, "g": [0, 2], "makefil": [0, 4], "under": 0, "docker": [0, 4], "complet": 0, "These": 0, "ar": [0, 1, 2, 3, 4, 5, 7, 8], "pip": [0, 4], "": [1, 2, 3, 5, 7, 8], "two": [1, 8], "wai": 1, "config": 1, "file": [1, 2, 6, 7], "automat": [1, 3, 8], "load": [1, 3, 7, 8], "pyrannc": [1, 2, 3, 4, 6, 7, 8], "rannc_conf": 1, "toml": [1, 6], "name": [1, 6, 7, 8], "item": 1, "lowercas": 1, "variabl": 1, "rannc_conf_dir": 1, "overwrit": 1, "rannc_": 1, "conf_item_nam": 1, "uppercas": 1, "For": [1, 4, 8], "exampl": [1, 2, 3, 5, 6, 8], "mem_margin": 1, "tabl": 1, "rannc_mem_margin": 1, "default": [1, 2, 7], "show_config_item": 1, "fals": [1, 7], "startup": 1, "true": [1, 2, 7, 8], "mem_limit_gb": 1, "memori": [1, 3, 7, 8], "limit": [1, 3, 8], "per": [1, 8], "devic": [1, 3, 4, 7, 8], "gb": 1, "posit": 1, "number": [1, 2, 8], "given": [1, 3, 7, 8], "1": [1, 3, 4], "margin": 1, "model": [1, 3, 5, 7], "partit": [1, 3, 7], "save_deploy": [1, 2, 7], "save": [1, 7], "deploy": [1, 2, 7], "load_deploy": [1, 2, 7], "deployment_fil": [1, 2], "tmp": 1, "rannc_deploy": 1, "partition_num": 1, "forc": 1, "parallel": [1, 2, 3, 7, 8], "valu": [1, 7], "min_pipelin": 1, "minimum": 1, "microbatch": 1, "pipelin": [1, 2, 7, 8], "max_pipelin": 1, "32": 1, "maximum": 1, "opt_param_factor": 1, "factor": 1, "estim": [1, 8], "usag": [1, 2, 3, 8], "optim": [1, 2, 3, 7, 8], "thi": [1, 2, 7, 8], "adam": [1, 3], "becaus": [1, 2, 3, 7, 8], "intern": [1, 3, 6], "data": [1, 8], "v": [1, 6], "whose": [1, 5], "size": [1, 2, 7, 8], "equival": [1, 8], "paramet": [1, 2, 3, 7, 8], "tensor": [1, 5, 7, 8], "trace_ev": 1, "trace": [1, 5, 8], "event": 1, "when": [1, 2, 7, 8], "significantli": 1, "degrad": 1, "perform": [1, 2, 3, 7], "event_trace_fil": 1, "rannc_event_trac": 1, "json": 1, "profile_by_acc": 1, "comput": [1, 3, 5, 7, 8], "time": [1, 2, 7, 8], "accumul": [1, 7], "finer": 1, "grain": 1, "subgraph": [1, 2, 7, 8], "drastic": 1, "reduc": 1, "patit": 1, "while": [1, 3], "accuraci": 1, "declin": 1, "sync_allreduc": 1, "synchron": [1, 7, 8], "allreduc": [1, 2, 7], "across": [1, 2, 8], "all": [1, 2, 7, 8], "stage": 1, "partitioning_dry_run_np": 1, "dry": 1, "determin": [1, 8], "ye": 2, "convert": [2, 8], "initi": [2, 3, 7], "pass": [2, 3, 7], "ranncmodul": [2, 3, 6, 7, 8], "use_amp_master_param": 2, "state_dict": [2, 7], "return": [2, 7, 8], "make": [2, 4], "sure": 2, "call": [2, 5, 7, 8], "from": [2, 3, 5, 7, 8], "rank": [2, 7, 8], "otherwis": 2, "would": 2, "block": [2, 7], "gather": [2, 8], "also": [2, 4], "modifi": [2, 8], "To": [2, 8], "collect": 2, "state": [2, 7, 8], "from_glob": 2, "after": [2, 5, 7], "load_state_dict": [2, 7], "keyword": [2, 5], "argument": [2, 8], "typic": 2, "As": [2, 5, 7], "implicitli": 2, "sum": [2, 8], "backward": [2, 3, 7], "prevent": 2, "delay_grad_allreduc": [2, 7], "specifi": [2, 4], "forward": [2, 3, 7], "step": [2, 3], "explicitli": [2, 7], "allreduce_grad": [2, 7], "By": [2, 8], "output": [2, 5, 7, 8], "greatli": 2, "program": [2, 4, 8], "similar": 2, "e": [2, 7], "onli": [2, 3, 4, 7, 8], "learn": [2, 3, 8], "rate": 2, "differ": [2, 8], "see": [2, 3, 8], "configur": [2, 3, 6, 8], "unsur": 2, "whether": 2, "continu": 2, "alreadi": 2, "fail": 2, "log": [2, 3], "level": [2, 6], "mlpartition": [2, 6], "dpstage": [2, 6, 8], "progress": 2, "option": [2, 8], "enabl": 2, "Then": [2, 8], "read": 2, "displai": 2, "show_deploy": [2, 7], "batch_siz": [2, 7, 8], "second": 2, "expect": [2, 8], "batch": [2, 5, 7, 8], "micro": [2, 7], "A": [2, 4, 8], "one": [2, 7, 8], "liner": 2, "api": [2, 3], "c": 2, "import": 2, "path_to_deployment_fil": 2, "64": [2, 8], "middlewar": 3, "train": [3, 7, 8], "veri": 3, "larg": 3, "scale": 3, "sinc": [3, 6, 8], "modern": 3, "often": [3, 7], "have": [3, 8], "billion": [3, 8], "thei": [3, 7, 8], "do": [3, 4, 8], "fit": [3, 8], "gpu": [3, 7, 8], "huge": 3, "multipl": [3, 7, 8], "compar": 3, "exist": 3, "framework": [3, 8], "includ": [3, 5, 7, 8], "megatron": 3, "lm": 3, "mesh": 3, "tensorflow": 3, "user": [3, 8], "implement": 3, "without": 3, "ani": 3, "modif": 3, "its": 3, "descript": 3, "In": [3, 8], "addit": 3, "basic": 3, "architectur": 3, "applic": [3, 7], "transform": 3, "base": 3, "code": 3, "simpl": [3, 8], "insert": [3, 8], "line": 3, "highlight": 3, "net": [3, 8], "defin": 3, "torch": [3, 4, 5, 7, 8], "move": [3, 7], "paramst": 3, "lr": [3, 8], "01": [3, 8], "wrap": 3, "loss": [3, 5], "input": [3, 7, 8], "updat": [3, 8], "abov": [3, 4, 8], "special": 3, "oper": 3, "distribut": [3, 4, 7, 8], "annot": 3, "our": [3, 8], "tutori": 3, "enlarg": 3, "version": [3, 4], "bert": 3, "resnet": 3, "subcompon": 3, "so": 3, "each": [3, 6, 7, 8], "high": 3, "throughput": [3, 8], "achiev": 3, "contrast": 3, "like": [3, 8], "columnparallellinear": 3, "rowparallellinear": 3, "hard": 3, "even": [3, 5], "expert": 3, "consid": 3, "commun": 3, "overhead": 3, "famili": 3, "we": 3, "confirm": 3, "approxim": 3, "100": [3, 8], "manual": [3, 4], "definit": 3, "idea": 3, "were": [3, 8], "publish": 3, "ipdp": 3, "2021": 3, "paper": 3, "algorithm": 3, "comparison": 3, "other": [3, 7, 8], "preprint": 3, "instal": [3, 8], "faq": 3, "build": 3, "sourc": 3, "graph": [3, 5, 7, 8], "deep": 3, "masahiro": 3, "tanaka": 3, "kenjiro": 3, "taura": 3, "toshihiro": 3, "hanawa": 3, "kentaro": 3, "torisawa": 3, "proceed": 3, "35th": 3, "ieee": 3, "symposium": 3, "pp": 3, "1004": 3, "1013": 3, "mai": [3, 8], "work": [4, 5, 7], "cpu": 4, "tpu": 4, "support": 4, "tool": [4, 8], "11": 4, "avail": [4, 8], "launch": 4, "openmpi": [4, 8], "v4": 4, "7": [4, 8], "libstd": 4, "glibcxx_3": 4, "21": 4, "5": 4, "current": 4, "v1": 4, "linux_x86_64": 4, "combin": 4, "3": 4, "8": [4, 8], "9": 4, "10": [4, 8], "command": [4, 8], "creat": 4, "new": [], "should": [4, 7], "cu": 4, "cuda_version_without_dot": 4, "cu113": 4, "f": [4, 6], "download": 4, "org": 4, "whl": 4, "torch_stabl": 4, "html": 4, "io": 4, "link": 4, "If": [4, 7, 8], "match": 4, "suitabl": 4, "wheel": 4, "although": 5, "ranncmodel": 5, "design": 5, "manner": [5, 8], "nn": [5, 7, 8], "modul": [5, 6, 7, 8], "produc": 5, "explain": 5, "document": 5, "doe": [5, 7, 8], "record": 5, "condit": 5, "branch": 5, "loop": 5, "howev": [5, 7], "jit": 5, "preserv": 5, "test_funct": 5, "test_simpl": [5, 8], "satisfi": 5, "mini": [5, 8], "first": [5, 7], "dimens": 5, "correspond": 5, "sampl": 5, "allow": 5, "scalar": 5, "spdlog": 6, "spdlog_setup": 6, "place": 6, "logger": 6, "associ": 6, "global_pattern": 6, "y": 6, "m": 6, "t": 6, "l": [6, 8], "n": 6, "sink": 6, "console_st": 6, "type": 6, "stdout_sink_st": 6, "stderr_st": 6, "color_stdout_sink_st": 6, "root": 6, "info": [6, 8], "ranncprocess": [6, 8], "graphlaunch": 6, "graphvaluestorag": 6, "graphutil": 6, "decompos": [6, 8], "decomposit": 6, "graphprofil": 6, "paramstorag": [6, 8], "graphconnector": 6, "torchdriv": 6, "ncclwrapper": 6, "class": [7, 8], "none": 7, "gather_input": 7, "enable_apex_amp": 7, "allreduce_amp_master_param": 7, "enable_zero": 7, "check_unused_valu": 7, "offload_param": 7, "hybrid": [7, 8], "apex": 7, "amp": 7, "gradient": [7, 8], "master": 7, "remov": 7, "redund": 7, "approach": 7, "deepspe": 7, "throw": 7, "except": 7, "unus": 7, "host": 7, "until": 7, "buffer": 7, "arg": 7, "kwarg": 7, "among": 7, "clip_grad_norm": 7, "max_grad_norm": 7, "clip": 7, "accord": 7, "norm": 7, "method": 7, "inst": 7, "util": [7, 8], "clip_grad_norm_": 7, "local": 7, "part": [7, 8], "calcul": 7, "them": [7, 8], "max": 7, "placement": 7, "control": 7, "enable_dropout": 7, "self": [7, 8], "_pyrannc": 7, "arg0": 7, "bool": 7, "eval": 7, "mode": 7, "evalu": 7, "get_param": 7, "int": [7, 8], "arg1": 7, "get_param_grad": 7, "origin": [7, 8], "named_buff": 7, "named_paramet": 7, "no_hook": 7, "amp_master_param": 7, "rank0_onli": 7, "hook": 7, "ignor": 7, "get": 7, "param": 7, "warn": 7, "cannot": [7, 8], "grad": 7, "undeploi": 7, "free": 7, "zero_grad": 7, "zero": [7, 8], "barrier": 7, "reach": 7, "clear": 7, "delai": 7, "soon": 7, "skip": 7, "get_rank": [7, 8], "comm_world": 7, "get_world_s": 7, "world": 7, "keep_graph": 7, "keep": 7, "flag": 7, "retain_graph": 7, "recreate_all_commun": 7, "debug": 7, "global": [7, 8], "sync_params_on_init": 7, "sync": 7, "aim": 7, "same": [7, 8], "take": [7, 8], "long": [7, 8], "random": 7, "seed": 7, "greatest": 8, "featur": 8, "written": 8, "unlik": 8, "ensur": 8, "shown": 8, "page": 8, "almost": 8, "opt": 8, "sgd": 8, "thu": 8, "declar": 8, "x": 8, "randn": 8, "hidden_s": 8, "requires_grad": 8, "out": 8, "randn_lik": 8, "sever": 8, "more": 8, "regard": 8, "detail": 8, "simpli": 8, "sy": 8, "def": 8, "__init__": 8, "hidden": 8, "layer": 8, "super": 8, "modulelist": 8, "linear": 8, "rang": 8, "argv": 8, "print": 8, "format": 8, "p": 8, "numel": 8, "target": 8, "finish": 8, "mpirun": 8, "begin": 8, "np": 8, "512": 8, "properli": 8, "mca": 8, "pml": 8, "ucx": 8, "btl": 8, "vader": 8, "tcp": 8, "openib": 8, "indic": 8, "alloc": 8, "node": 8, "equal": 8, "than": 8, "eight": 8, "nvidia": 8, "a100": 8, "40gb": 8, "coll": 8, "hcoll": 8, "start": 8, "gpunode001": 8, "assign": 8, "worker": 8, "device0": 8, "device1": 8, "2626560": 8, "ir": 8, "assum": 8, "128": 8, "ml_part": 8, "38255689728": 8, "use_amp": 8, "merge_0_9": 8, "repl": 8, "fwd_time": 8, "4722": 8, "bwd_time": 8, "24237": 8, "ar_tim": 8, "978": 8, "in_siz": 8, "131072": 8, "out_siz": 8, "fp32param_s": 8, "10506240": 8, "fp16param_s": 8, "total_mem": 8, "54759424": 8, "fwd": 8, "bwd": 8, "33353728": 8, "21012480": 8, "comm": 8, "393216": 8, "rout": 8, "verif": 8, "readi": 8, "rank0": 8, "rank1": 8, "result": 8, "wa": 8, "replic": 8, "singl": 8, "effect": 8, "distributedsampl": 8, "5000": 8, "respect": 8, "exce": 8, "10gb": 8, "20gb": 8, "let": 8, "2500500000": 8, "merge_0_4": 8, "27516": 8, "126756": 8, "437809": 8, "2560000": 8, "4700940000": 8, "23707792544": 8, "14298232544": 8, "9401880000": 8, "7680000": 8, "merge_5_9": 8, "31228": 8, "153762": 8, "493699": 8, "5301060000": 8, "26732209376": 8, "16122409376": 8, "10602120000": 8, "6": 8, "It": 8, "took": 8, "around": 8, "five": 8, "minut": 8, "replica": 8, "practic": 8, "pytoch": 4}, "objects": {"": [[7, 0, 0, "-", "pyrannc"]], "pyrannc": [[7, 1, 1, "", "RaNNCModule"], [7, 3, 1, "", "barrier"], [7, 3, 1, "", "clear"], [7, 3, 1, "", "delay_grad_allreduce"], [7, 3, 1, "", "get_rank"], [7, 3, 1, "", "get_world_size"], [7, 3, 1, "", "keep_graph"], [7, 3, 1, "", "recreate_all_communicators"], [7, 3, 1, "", "show_deployment"], [7, 3, 1, "", "sync_params_on_init"]], "pyrannc.RaNNCModule": [[7, 2, 1, "", "buffers"], [7, 2, 1, "", "clip_grad_norm"], [7, 2, 1, "", "cuda"], [7, 2, 1, "", "enable_dropout"], [7, 2, 1, "", "eval"], [7, 2, 1, "", "get_param"], [7, 2, 1, "", "get_param_grad"], [7, 2, 1, "", "load_state_dict"], [7, 2, 1, "", "named_buffers"], [7, 2, 1, "", "named_parameters"], [7, 2, 1, "", "parameters"], [7, 2, 1, "", "save_deployment"], [7, 2, 1, "", "state_dict"], [7, 2, 1, "", "to"], [7, 2, 1, "", "train"], [7, 2, 1, "", "undeploy"], [7, 2, 1, "", "zero_grad"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method", "3": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"], "3": ["py", "function", "Python function"]}, "titleterms": {"build": 0, "from": 0, "sourc": 0, "compil": 0, "version": 0, "instal": [0, 4], "variabl": 0, "configur": [0, 1], "faq": 2, "doe": 2, "rannc": [2, 3, 8], "work": 2, "apex": 2, "amp": 2, "how": 2, "can": 2, "i": 2, "save": 2, "load": 2, "modul": 2, "us": [2, 8], "gradient": 2, "accumul": 2, "my": 2, "model": [2, 8], "take": 2, "too": 2, "long": 2, "befor": 2, "partit": [2, 8], "determin": 2, "check": 2, "result": 2, "rapid": 3, "neural": 3, "network": 3, "connector": 3, "content": 3, "refer": [3, 7], "prerequisit": 4, "limit": 5, "control": 5, "construct": 5, "argument": 5, "return": 5, "valu": 5, "log": 6, "api": 7, "tutori": 8, "step": 8, "0": 8, "set": 8, "up": 8, "environ": 8, "1": 8, "import": 8, "2": 8, "wrap": 8, "your": 8, "3": 8, "run": 8, "forward": 8, "backward": 8, "pass": 8, "4": 8, "launch": 8, "small": 8, "5": 8, "veri": 8, "larg": 8}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 57}, "alltitles": {"Building from source": [[0, "building-from-source"]], "Compiler version": [[0, "compiler-version"]], "Build and Install": [[0, "build-and-install"]], "Variables for building configurations": [[0, "id1"]], "Configurations": [[1, "configurations"], [1, "id1"]], "FAQs": [[2, "faqs"]], "Does RaNNC work with Apex AMP?": [[2, "does-rannc-work-with-apex-amp"]], "How can I save/load a RaNNC module?": [[2, "how-can-i-save-load-a-rannc-module"]], "Can I use gradient accumulation?": [[2, "can-i-use-gradient-accumulation"]], "My model takes too long before partitioning is determined": [[2, "my-model-takes-too-long-before-partitioning-is-determined"]], "How can I check partitioning results?": [[2, "how-can-i-check-partitioning-results"]], "RaNNC (Rapid Neural Network Connector)": [[3, "rannc-rapid-neural-network-connector"]], "Contents:": [[3, null]], "Reference": [[3, "reference"]], "Limitations": [[5, "limitations"]], "Control constructs": [[5, "control-constructs"]], "Arguments and return values": [[5, "arguments-and-return-values"]], "Logging": [[6, "logging"]], "API references": [[7, "module-pyrannc"]], "Tutorial": [[8, "tutorial"]], "Steps to use RaNNC": [[8, "steps-to-use-rannc"]], "0. Set up environment": [[8, "set-up-environment"]], "1. Import RaNNC": [[8, "import-rannc"]], "2. Wrap your model": [[8, "wrap-your-model"]], "3. Run forward/backward passes": [[8, "run-forward-backward-passes"]], "4. Launch (with a small model)": [[8, "launch-with-a-small-model"]], "5. Model partitioning for very large models": [[8, "model-partitioning-for-very-large-models"]], "Installation": [[4, "installation"], [4, "id1"]], "Prerequisites": [[4, "prerequisites"]]}, "indexentries": {}})