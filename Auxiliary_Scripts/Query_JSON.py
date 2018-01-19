import urllib
import urllib2
import json
import glob
import numpy as np

# Append the service name to this base URL, eg 'con', 'obs', etc.
BASEURL = 'http://mwa-metadata01.pawsey.org.au/metadata/'

obslist_path = '/Users/mike_e_dubs/MWA/Obs_Lists/Diffuse_2015_GP_10s_Autos_RFI_Free.txt'
with open(obslist_path) as f:
    obslist = f.read().split("\n")

obslist.remove('')

obslist = np.array(obslist).astype(int).tolist()


# Function to call a JSON web service and return a dictionary:


def getmeta(service='obs', params=None):
    """Given a JSON web service ('obs', find, or 'con') and a set of
     parameters as a Python dictionary, return a Python dictionary
     containing the result.
    """
    if params:
        data = urllib.urlencode(params)
    else:
        data = ''
    # Validate the service name
    if service.strip().lower() in ['obs', 'find', 'con', 'temps']:
        service = service.strip().lower()
    else:
        print "invalid service name: %s" % service
        return
    # Get the data
    try:
        result = json.load(urllib2.urlopen(BASEURL + service + '?' + data))
    except urllib2.HTTPError as error:
        print "HTTP error: code=%d, :\n %s" % (error.code, error.read())
        return
    except urllib2.URLError as error:
        print "URL or network error: %s" % error.reason
        return
    # Return the result dictionary
    return result


# Get and print observation info:

temp_list = []

for obs in obslist:
    # Some example query data:
    starttime = obs

# obsinfo1 = getmeta(service='obs', params={'obs_id': starttime})

# coninfo1 = getmeta(service='con', params={'obs_id': starttime})

    temps = getmeta(service='temps', params={'starttime': starttime})
    temp_list.append(np.mean([temps[item][3] for item in temps]))

np.save('/Users/mike_e_dubs/MWA/Temperatures/Diffuse_2015_12s_Autos/Metadata/bftemps.npy',
        np.array(temp_list))
print(temp_list)
