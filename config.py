from __future__ import division, print_function, absolute_import
import argparse
import json


def named_choices(choices):
    def convert(val):
        if val not in choices:
            raise Exception('%s is not a recognized '
                            'choice (choices are %s)'
                            % (val, ', '.join(choices.keys())))
        return choices[val]
    return convert


class Config(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Train instance task using dataset interface")
        self.parser.add_argument('--config', default=None,
            type=str,
            help="Path to a JSON file containing configuration info. Any " \
                 "configurations loaded from this file are superseded by " \
                 "configurations passed from the command line.")
        self.fields = []
        self.required_fields = []

        self._reserved = ['config', 'description']
        self._default_values = {}
        self._types = {}

    def add(self, field, type, help,
            default=None, required=False,
            action='store'):
        def _assert(cond, mesg):
            if not cond:
                raise Exception("Error in defining flag %s: %s" % (field, mesg))
        _assert(field not in self._reserved, "flag name reserved!")
        _assert(field not in self.fields, "already defined!")

        if type is bool:
            if default is None:
                default = False
            self.parser.add_argument(
                '--' + field, default=None,
                help=help, action='store_true')
        else:
            self.parser.add_argument(
                '--' + field, default=None, type=type,
                help=help, action=action)

        self.fields.append(field)
        self._types[field] = type

        if default is not None:
            _assert(not required, "default doesn't make sense " \
                    "when flag is required!")
            self._default_values[field] = type(default)
        if required:
            self.required_fields.append(field)

    def parse_config_file(self, config_str):
        if config_str is None:
            return {}

        parts = config_str.split(':')
        assert len(parts) <= 2
        if len(parts) < 2:
            parts.append(None)
        path, config_name = parts

        def strip_comments(s):
            # Quick-and-dirty way to strip comments. Should work for our
            # purposes.
            lines = s.split('\n')
            lines = filter(lambda x: not x.strip().startswith('//'), lines)
            return '\n'.join(lines)

        f = open(path)
        json_str = strip_comments(f.read())
        json_dict = json.loads(json_str)
        if config_name is not None:
            if config_name not in json_dict:
                raise Exception("Could not find configuration called '%s' "
                                "in file '%s'" % (config_name, path))
            json_dict = json_dict[config_name]
        return json_dict

    def parse_args(self):
        args = self.parser.parse_args()
        file_cfg = self.parse_config_file(args.config)

        # Configuration priority:
        # 1. Explicit command line values
        # 2. Config file values
        # 3. Default values
        for field in self.fields:
            cmd_val = getattr(args, field)
            if cmd_val is not None:
                continue

            if field in file_cfg:
                value = self._types[field](file_cfg[field])
                setattr(args, field, value)
            elif field in self._default_values:
                setattr(args, field, self._default_values[field])

        for field in self.required_fields:
            if getattr(args, field) is None:
                raise Exception("Missing required argument %s" % field)
        return args
