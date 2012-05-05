"""
util tests

"""
import os
import pkg_resources
from mock import patch, sentinel
from nose.tools import eq_
from tests.path import Path
from pip.util import path_in_dir, egg_link_path


class Tests_EgglinkPath:
    "util.egg_link_path() tests"
    dist = pkg_resources.get_distribution('pip') #doesn't have to be pip
    user_site = Path('USER_SITE')
    site_packages = Path('SITE_PACKAGES')
    user_site_egglink = Path('USER_SITE','pip.egg-link')
    site_packages_egglink = Path('SITE_PACKAGES','pip.egg-link')

    def isFileUserSite(self,egglink):
        if egglink==self.user_site_egglink:
            return True

    def isFileSitePackages(self,egglink):
        if egglink==self.site_packages_egglink:
            return True        

    #########################
    ## egglink in usersite ##
    #########################
    @patch('pip.util.site_packages', Path('SITE_PACKAGES'))
    @patch('os.path.isfile')
    @patch('pip.util.running_under_virtualenv')
    @patch('pip.util.virtualenv_no_global')
    @patch('pip.util.user_site')
    def test_egglink_in_usersite_notvenv(self,mock_user_site, mock_virtualenv_no_global, mock_running_under_virtualenv, mock_isfile):
        mock_user_site.return_value = self.user_site
        mock_virtualenv_no_global.return_value = False
        mock_running_under_virtualenv.return_value = False
        mock_isfile.side_effect = self.isFileUserSite
        eq_(egg_link_path(self.dist), self.user_site_egglink)

    @patch('pip.util.site_packages', Path('SITE_PACKAGES'))
    @patch('os.path.isfile')
    @patch('pip.util.running_under_virtualenv')
    @patch('pip.util.virtualenv_no_global')
    @patch('pip.util.user_site')
    def test_egglink_in_usersite_venv_noglobal(self,mock_user_site, mock_virtualenv_no_global, mock_running_under_virtualenv, mock_isfile):
        mock_user_site.return_value = self.user_site
        mock_virtualenv_no_global.return_value = True
        mock_running_under_virtualenv.return_value = True
        mock_isfile.side_effect = self.isFileUserSite
        eq_(egg_link_path(self.dist), None)

    @patch('pip.util.site_packages', Path('SITE_PACKAGES'))
    @patch('os.path.isfile')
    @patch('pip.util.running_under_virtualenv')
    @patch('pip.util.virtualenv_no_global')
    @patch('pip.util.user_site')
    def test_egglink_in_usersite_venv_global(self,mock_user_site, mock_virtualenv_no_global, mock_running_under_virtualenv, mock_isfile):
        mock_user_site.return_value = self.user_site
        mock_virtualenv_no_global.return_value = False
        mock_running_under_virtualenv.return_value = True
        mock_isfile.side_effect = self.isFileUserSite
        eq_(egg_link_path(self.dist), self.user_site_egglink)

    #########################
    ## egglink in sitepkgs ##
    #########################
    @patch('pip.util.site_packages', Path('SITE_PACKAGES'))
    @patch('os.path.isfile')
    @patch('pip.util.running_under_virtualenv')
    @patch('pip.util.virtualenv_no_global')
    @patch('pip.util.user_site')
    def test_egglink_in_sitepkgs_notvenv(self,mock_user_site, mock_virtualenv_no_global, mock_running_under_virtualenv, mock_isfile):
        mock_user_site.return_value = self.user_site
        mock_virtualenv_no_global.return_value = False
        mock_running_under_virtualenv.return_value = False
        mock_isfile.side_effect = self.isFileSitePackages
        eq_(egg_link_path(self.dist), self.site_packages_egglink)

    @patch('pip.util.site_packages', Path('SITE_PACKAGES'))
    @patch('os.path.isfile')
    @patch('pip.util.running_under_virtualenv')
    @patch('pip.util.virtualenv_no_global')
    @patch('pip.util.user_site')
    def test_egglink_in_sitepkgs__venv_noglobal(self,mock_user_site, mock_virtualenv_no_global, mock_running_under_virtualenv, mock_isfile):
        mock_user_site.return_value = self.user_site
        mock_virtualenv_no_global.return_value = True
        mock_running_under_virtualenv.return_value = True
        mock_isfile.side_effect = self.isFileSitePackages
        eq_(egg_link_path(self.dist), self.site_packages_egglink)

    @patch('pip.util.site_packages', Path('SITE_PACKAGES'))
    @patch('os.path.isfile')
    @patch('pip.util.running_under_virtualenv')
    @patch('pip.util.virtualenv_no_global')
    @patch('pip.util.user_site')
    def test_egglink_in_sitepkgs__venv_2global(self,mock_user_site, mock_virtualenv_no_global, mock_running_under_virtualenv, mock_isfile):
        mock_user_site.return_value = self.user_site
        mock_virtualenv_no_global.return_value = True
        mock_running_under_virtualenv.return_value = True
        mock_isfile.side_effect = self.isFileSitePackages
        eq_(egg_link_path(self.dist), self.site_packages_egglink)


    ####################################
    ## egglink in usersite & sitepkgs ##
    ####################################
    @patch('pip.util.site_packages', Path('SITE_PACKAGES'))
    @patch('os.path.isfile')
    @patch('pip.util.running_under_virtualenv')
    @patch('pip.util.virtualenv_no_global')
    @patch('pip.util.user_site')
    def test_egglink_in_both_notvenv(self,mock_user_site, mock_virtualenv_no_global, mock_running_under_virtualenv, mock_isfile):
        mock_user_site.return_value = self.user_site
        mock_virtualenv_no_global.return_value = False
        mock_running_under_virtualenv.return_value = False
        mock_isfile.return_value = True
        eq_(egg_link_path(self.dist), self.user_site_egglink)

    @patch('pip.util.site_packages', Path('SITE_PACKAGES'))
    @patch('os.path.isfile')
    @patch('pip.util.running_under_virtualenv')
    @patch('pip.util.virtualenv_no_global')
    @patch('pip.util.user_site')
    def test_egglink_in_both_venv_noglobal(self,mock_user_site, mock_virtualenv_no_global, mock_running_under_virtualenv, mock_isfile):
        mock_user_site.return_value = self.user_site
        mock_virtualenv_no_global.return_value = True
        mock_running_under_virtualenv.return_value = True
        mock_isfile.return_value = True
        eq_(egg_link_path(self.dist), self.site_packages_egglink)

    @patch('pip.util.site_packages', Path('SITE_PACKAGES'))
    @patch('os.path.isfile')
    @patch('pip.util.running_under_virtualenv')
    @patch('pip.util.virtualenv_no_global')
    @patch('pip.util.user_site')
    def test_egglink_in_both_venv_global(self,mock_user_site, mock_virtualenv_no_global, mock_running_under_virtualenv, mock_isfile):
        mock_user_site.return_value = self.user_site
        mock_virtualenv_no_global.return_value = False
        mock_running_under_virtualenv.return_value = True
        mock_isfile.return_value = True
        eq_(egg_link_path(self.dist), self.user_site_egglink)


    ################
    ## no egglink ##
    ################
    @patch('pip.util.site_packages', Path('SITE_PACKAGES'))
    @patch('os.path.isfile')
    @patch('pip.util.running_under_virtualenv')
    @patch('pip.util.virtualenv_no_global')
    @patch('pip.util.user_site')
    def test_noegglink_in_sitepkgs_notvenv(self,mock_user_site, mock_virtualenv_no_global, mock_running_under_virtualenv, mock_isfile):
        mock_user_site.return_value = self.user_site
        mock_virtualenv_no_global.return_value = False
        mock_running_under_virtualenv.return_value = False
        mock_isfile.return_value = False
        eq_(egg_link_path(self.dist), None)

    @patch('pip.util.site_packages', Path('SITE_PACKAGES'))
    @patch('os.path.isfile')
    @patch('pip.util.running_under_virtualenv')
    @patch('pip.util.virtualenv_no_global')
    @patch('pip.util.user_site')
    def test_noegglink_in_sitepkgs__venv_noglobal(self,mock_user_site, mock_virtualenv_no_global, mock_running_under_virtualenv, mock_isfile):
        mock_user_site.return_value = self.user_site
        mock_virtualenv_no_global.return_value = True
        mock_running_under_virtualenv.return_value = True
        mock_isfile.return_value = False
        eq_(egg_link_path(self.dist), None)

    @patch('pip.util.site_packages', Path('SITE_PACKAGES'))
    @patch('os.path.isfile')
    @patch('pip.util.running_under_virtualenv')
    @patch('pip.util.virtualenv_no_global')
    @patch('pip.util.user_site')
    def test_noegglink_in_sitepkgs__venv_2global(self,mock_user_site, mock_virtualenv_no_global, mock_running_under_virtualenv, mock_isfile):
        mock_user_site.return_value = self.user_site
        mock_virtualenv_no_global.return_value = True
        mock_running_under_virtualenv.return_value = True
        mock_isfile.return_value = False
        eq_(egg_link_path(self.dist), None)


class Tests_PathInDir:
    "util.path_in_dir() tests"

    dir1 = Path('dir1')
    dir2 = Path('dir2')
    dir2_sep = dir2 + os.path.sep
    dir1_cat_dir2 = dir1 + dir2
    dir2_in_dir1 = dir1 / dir2    

    def path_in_dir(self,p1, p2, v):
        eq_(path_in_dir(p1,p2), v, "%s in %s is not %s" %(p1, p2, v))

    def test_dir_in_dir(self):
        "dir1/dir2 is in dir1"
        self.path_in_dir(self.dir2_in_dir1, self.dir1, True)

    def test_dir_notin_dir(self):
        "dir2 is not in dir1"
        self.path_in_dir(self.dir2, self.dir1, False)

    def test_dircat_notin_dir(self):
        "dir1dir2 is not in dir1"
        self.path_in_dir(self.dir1_cat_dir2, self.dir1, False)

    def test_dir_eq_dir(self):
        "dir1 is 'in' dir1"
        self.path_in_dir(self.dir1, self.dir1, True)

    def test_dir_eq_dirsep(self):
        "dir2 is 'in' dir2/"
        self.path_in_dir(self.dir2, self.dir2_sep, True)



