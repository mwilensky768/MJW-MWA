import urllib.request
import json
import argparse
import yaml
import time

# Append the service name to this base URL, e.g. 'con', 'obs', etc.
BASEURL = 'http://ws.mwatelescope.org/'


def getmeta(servicetype='metadata', service='temps', params=None):
    """Given a JSON web servicetype ('observation' or 'metadata'), a service name (eg 'obs', find, or 'con')
       and a set of parameters as a Python dictionary, return a Python dictionary containing the result.
    """
    if params:
        # Turn the dictionary into a string with encoded 'name=value' pairs
        data = urllib.parse.urlencode(params)
    else:
        data = ''

    # Get the data
    try:
        result = json.load(urllib.request.urlopen(BASEURL + servicetype + '/' + service + '?' + data))
    except urllib.error.HTTPError as err:
        print("HTTP error from server: code=%d, response:\n %s" % (err.code, err.read()))
        return
    except urllib.error.URLError as err:
        print("URL or network error: %s" % err.reason)
        return

    # Return the result dictionary
    return result


def make_obslist(obsfile):
    # due to the newline character, raw string is needed `r"""`
    r"""
    Makes a python list from a text file whose lines are separated by "\\n"

    Args:
        obsfile: A text file with an obsid on each line

    Returns:
        obslist: A list whose entries are obsids
    """
    with open(obsfile) as f:
        obslist = f.read().split("\n")
    while '' in obslist:
        obslist.remove('')
    obslist.sort()
    return(obslist)


def make_obsfile(obslist, outpath):
    """
    Makes a text file from a list of obsids

    Args:
        obslist: A list of obsids
        outpath: The filename to write to
    """
    with open(outpath, 'w') as f:
        for obs in obslist:
            f.write(f"{obs}\n")


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--obsfile", help="The obsids to get the temperature data for.")
parser.add_argument("-o", "--outfile", help="The output directory to put the file in, with prefix")
args = parser.parse_args()

obslist = make_obslist(args.obsfile)
temp_dict = {}
for ind, obs in enumerate(obslist):
    params = {"obsid": obs, "dictformat": 1}
    server_dict = getmeta(params=params)
    first_key = [key for key in server_dict][0]
    temp_dict[obs] = server_dict[first_key]
    if ind % 100 == 0:
        print(f"sleeping on ind {ind}")
        time.sleep(10)

outpath = f"{args.outfile}_temp_dict.yml"
with open(outpath, 'w') as outfile:
    yaml.safe_dump(temp_dict, outfile)
