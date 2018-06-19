from __future__ import absolute_import

import json
import logging
import os
from email.parser import FeedParser  # type: ignore

from pip._vendor import pkg_resources
from pip._vendor.packaging.utils import canonicalize_name

from pip._internal.basecommand import Command
from pip._internal.status_codes import ERROR, SUCCESS

logger = logging.getLogger(__name__)


class ShowCommand(Command):
    """Show information about one or more installed packages."""
    name = 'show'
    usage = """
      %prog [options] <package> ..."""
    summary = 'Show information about installed packages.'
    ignore_require_venv = True

    def __init__(self, *args, **kw):
        super(ShowCommand, self).__init__(*args, **kw)
        self.cmd_opts.add_option(
            '--json',
            dest='json',
            action='store_true',
            default=False,
            help='Output the result as json.')

        self.cmd_opts.add_option(
            '-f', '--files',
            dest='files',
            action='store_true',
            default=False,
            help='Show the full list of installed files for each package.')

        self.parser.insert_option_group(0, self.cmd_opts)

    def run(self, options, args):
        if not args:
            logger.warning('ERROR: Please provide a package name or names.')
            return ERROR
        query = args

        results = search_packages_info(query)
        if options.json:
            if not print_results_json(results, options):
                return ERROR
        else:
            if not print_results(results, options):
                return ERROR
        return SUCCESS


def search_packages_info(query):
    """
    Gather details from installed distributions. Print distribution name,
    version, location, and installed files. Installed files requires a
    pip generated 'installed-files.txt' in the distributions '.egg-info'
    directory.
    """
    installed = {}
    for p in pkg_resources.working_set:
        installed[canonicalize_name(p.project_name)] = p

    query_names = [canonicalize_name(name) for name in query]

    for dist in [installed[pkg] for pkg in query_names if pkg in installed]:
        package = {
            'name': dist.project_name,
            'version': dist.version,
            'location': dist.location,
            'requires': [dep.project_name for dep in dist.requires()],
        }
        file_list = None
        metadata = None
        if isinstance(dist, pkg_resources.DistInfoDistribution):
            # RECORDs should be part of .dist-info metadatas
            if dist.has_metadata('RECORD'):
                lines = dist.get_metadata_lines('RECORD')
                paths = [l.split(',')[0] for l in lines]
                paths = [os.path.join(dist.location, p) for p in paths]
                file_list = [os.path.relpath(p, dist.location) for p in paths]

            if dist.has_metadata('METADATA'):
                metadata = dist.get_metadata('METADATA')
        else:
            # Otherwise use pip's log for .egg-info's
            if dist.has_metadata('installed-files.txt'):
                paths = dist.get_metadata_lines('installed-files.txt')
                paths = [os.path.join(dist.egg_info, p) for p in paths]
                file_list = [os.path.relpath(p, dist.location) for p in paths]

            if dist.has_metadata('PKG-INFO'):
                metadata = dist.get_metadata('PKG-INFO')

        if dist.has_metadata('entry_points.txt'):
            entry_points = dist.get_metadata_lines('entry_points.txt')
            package['entry_points'] = entry_points

        if dist.has_metadata('INSTALLER'):
            for line in dist.get_metadata_lines('INSTALLER'):
                if line.strip():
                    package['installer'] = line.strip()
                    break

        # @todo: Should pkg_resources.Distribution have a
        # `get_pkg_info` method?
        feed_parser = FeedParser()
        feed_parser.feed(metadata)
        pkg_info_dict = feed_parser.close()
        for key in ('metadata-version', 'summary',
                    'home-page', 'author', 'author-email', 'license'):
            package[key] = pkg_info_dict.get(key)

        # It looks like FeedParser cannot deal with repeated headers
        classifiers = []
        for line in metadata.splitlines():
            if line.startswith('Classifier: '):
                classifiers.append(line[len('Classifier: '):])
        package['classifiers'] = classifiers

        if file_list:
            package['files'] = sorted(file_list)
        yield package


def print_results(distributions, options):
    """
    Print the informations from installed distributions found.
    """
    results_printed = False
    for i, dist in enumerate(distributions):
        results_printed = True
        if i > 0:
            logger.info("---")

        name = dist.get('name', '')
        required_by = [
            pkg.project_name for pkg in pkg_resources.working_set
            if name in [required.name for required in pkg.requires()]
        ]

        logger.info("Name: %s", name)
        logger.info("Version: %s", dist.get('version', ''))
        logger.info("Summary: %s", dist.get('summary', ''))
        logger.info("Home-page: %s", dist.get('home-page', ''))
        logger.info("Author: %s", dist.get('author', ''))
        logger.info("Author-email: %s", dist.get('author-email', ''))
        logger.info("License: %s", dist.get('license', ''))
        logger.info("Location: %s", dist.get('location', ''))
        logger.info("Requires: %s", ', '.join(dist.get('requires', [])))
        logger.info("Required-by: %s", ', '.join(required_by))

        if options.verbose:
            logger.info("Metadata-Version: %s",
                        dist.get('metadata-version', ''))
            logger.info("Installer: %s", dist.get('installer', ''))
            logger.info("Classifiers:")
            for classifier in dist.get('classifiers', []):
                logger.info("  %s", classifier)
            logger.info("Entry-points:")
            for entry in dist.get('entry_points', []):
                logger.info("  %s", entry.strip())
        if options.files:
            logger.info("Files:")
            for line in dist.get('files', []):
                logger.info("  %s", line.strip())
            if "files" not in dist:
                logger.info("Cannot locate installed-files.txt")

    return results_printed

def print_results_json(distributions, options):
    """
    Print the informations from installed distributions found in json.
    """
    results_printed = False
    json_out = []
    for i, dist in enumerate(distributions):
        results_printed = True

        name = dist.get('name', '')
        required_by = [
            pkg.project_name for pkg in pkg_resources.working_set
            if name in [required.name for required in pkg.requires()]
        ]

        json_dict_temp = {
            'name': dist.get('name', ''),
            'version': dist.get('version', ''),
            'summary': dist.get('summary', ''),
            'home-page': ('home-page', ''),
            'author': dist.get('author', ''),
            'author-email': dist.get('author-email', ''),
            'license': dist.get('license', ''),
            'location': dist.get('location', ''),
            'requires': ', '.join(dist.get('requires', [])),
            'required-by': ', '.join(required_by)
        }

        if options.verbose:
            metadata_version = dist.get('metadata-version', '')
            json_dict_temp['metadata-version'] = metadata_version
            json_dict_temp['installer'] = dist.get('installer', '')
            json_dict_temp['classifiers'] = dist.get('classifiers', [])
            json_dict_temp['entry-points'] = dist.get('entry_points', [])

        if options.files:
            json_files_out = []
            for line in dist.get('files', []):
                json_files_out.append(line.strip())
            if "files" not in dist:
                json_files_out.append("Cannot locate installed-files.txt")
            json_dict_temp['files'] = json_files_out

        json_out.append(json_dict_temp)
    
        logger.info(json.dumps(json_out))
        return results_printed
