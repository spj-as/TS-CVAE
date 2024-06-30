from argparse import ArgumentParser


def cli() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "option",
        type=str,
        choices=[
            "preprocess",
            "preprocess_bsk",
            "dagnet",
            "dagnet_wo_cond",
            "dagnet_w_all",
            "dagnet_wo_posemb",
            "dagnet_wo_graph",
            "dagnet_wo_graph_bsk"
        ],
        help="option",
    )
    return parser


def main():
    parser = cli()
    args, _ = parser.parse_known_args()

    if args.option == "preprocess":
        from runs.run_preprocess import preprocess_cli, preprocess_main

        args = preprocess_cli(parser).parse_args()
        preprocess_main(args)

    elif args.option == "preprocess_bsk":
        from runs.run_preprocess_bsk import preprocess_cli, preprocess_main

        args = preprocess_cli(parser).parse_args()
        preprocess_main(args)
    elif args.option == "dagnet":
        from runs.run_dagnet import dagnet_cli, dagnet_main

    elif args.option == "dagnet_wo_cond":
        from runs.run_dagnet_wo_cond import dagnet_cli, dagnet_main  
    
    elif args.option == "dagnet_w_all":
        from runs.run_dagnet_w_all import dagnet_cli, dagnet_main    

    elif args.option == "dagnet_wo_posemb":
        from runs.run_dagnet_wo_posemb import dagnet_cli, dagnet_main    

    elif args.option == "dagnet_wo_graph":
        from runs.run_dagnet_wo_graph import dagnet_cli, dagnet_main    

        args = dagnet_cli(parser).parse_args()
        dagnet_main(args)    
    elif args.option == "dagnet_wo_graph_bsk":
        from runs.run_dagnet_wo_graph_bsk import dagnet_cli, dagnet_main    

        args = dagnet_cli(parser).parse_args()
        dagnet_main(args)    


if __name__ == "__main__":
    main()
