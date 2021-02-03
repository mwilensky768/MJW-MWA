import urllib.request
import json
import argparse
from SSINS.util import make_obsfile

# Append the service name to this base URL, e.g. 'con', 'obs', etc.
BASEURL = 'http://ws.mwatelescope.org/'


def getmeta(servicetype='metadata', service='find', params=None):
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
parser.add_argument('-p', '--pointing', help="The pointing you want.", required=True)
parser.add_argument('--mintime', help="minimum obsid", required=True)
parser.add_argument('--maxtime', help="maximum obsid", required=True)
parser.add_argument('-o', '--outfile',
                    help="The output directory and a prefix. Will have _'<pointing>.txt' appended",
                    required=True)
args = parser.parse_args()

pnt_dict = {"minus_two": "3", "minus_one": "1", "zenith": 0, "plus_one": 2, "plus_two": 4}
gridpoint = pnt_dict[args.pointing]

params = {"minra": 0, "maxra": 10, "mindec": -32, "maxdec": -22,
          "projectid": "G0009", "anychan": 140, "gridpoint": gridpoint,
          "mintime": args.mintime, "maxtime": args.maxtime}


faultdict1 = getmeta(params=params)

params.update({"minra": 350, "maxra": 360})

faultdict2 = getmeta(params=params)

obslist = []
for item in faultdict1:
    obslist.append(item[0])

for item in faultdict2:
    obslist.append(item[0])

outpath = f"{args.outfile}_{args.pointing}.txt"
obslist = sorted(obslist)
make_obsfile(obslist, outpath)
