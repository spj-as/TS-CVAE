from argparse import ArgumentParser


def cli() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "option",
        type=str,
        choices=[
            "preprocess",
            "tscvae",
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


    elif args.option == "tscvae":
        from runs.run_tscvae import tscvae_cli, tscvae_main    

        args = tscvae_cli(parser).parse_args()
        tscvae_main(args)    
   


if __name__ == "__main__":
    main()
