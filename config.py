import configparser
import argparse

class App:
    file_path = None
    _conf = None

    @staticmethod
    def config():
        if App.file_path is None:
            print('Checking for alternative file path to .ini file')
            parser = argparse.ArgumentParser(
                description='Create predictions for different types of embeddings with and without evolutionary '
                            'information')
            parser.add_argument('-f',
                                default="ev_embeddings.ini",
                                type=str,
                                help='File path to .ini file if it isn\'t ev_embeddings.ini',
                                metavar="file_path")
            args = parser.parse_args()
            App.file_path = args.f
        if App._conf is None:
            App._conf = configparser.ConfigParser()
            App._conf.read(App.file_path)

        return App._conf
