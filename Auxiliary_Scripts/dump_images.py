"""
dumpimages.py
    Downloads all the images on the supplied URL, and saves them to the
    specified output file ("/test/" by default)

Usage:
    python dumpimages.py http://example.com/ [output]
"""

from BeautifulSoup import BeautifulSoup as bs
import urlparse
from urllib2 import urlopen
from urllib import urlretrieve
import os
import sys
import time


def main(url, out_folder="/test/"):
    """Downloads all the images at 'url' to /test/"""
    soup = bs(urlopen(url))
    parsed = list(urlparse.urlparse(url))
    time.sleep(60)

    for image in soup.findAll("img"):
        print "Image: %(src)s" % image
        filename = image["src"].split("/")[-1]
        parsed[2] = image["src"]
        outpath = os.path.join(out_folder, filename)
        if image["src"].lower().startswith("http"):
            urlretrieve(image["src"], outpath)
        else:
            urlretrieve(urlparse.urlunparse(parsed), outpath)


def _usage():
    print "usage: python dumpimages.py http://example.com [outpath]"


if __name__ == "__main__":
    url = sys.argv[-1]
    out_folder = "/Users/mike_e_dubs/MWA/Catalogs/Golden_Set_8s_Autos/Metadata_Images/"
    if not url.lower().startswith("http"):
        out_folder = sys.argv[-1]
        url = sys.argv[-2]
        if not url.lower().startswith("http"):
            _usage()
            sys.exit(-1)
    main(url, out_folder)
