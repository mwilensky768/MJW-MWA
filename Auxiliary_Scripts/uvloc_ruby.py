#!/usr/bin/python

#Script that locates uvfits files for a list of obsids.
#When keyword delete is set, the script deletes the empty directories it encounters.

import os
from astropy.time import Time

def main():

    obsfile_name = "/nfs/eor-00/h1/mwilensk/Diffuse_2015_8s_Autos/diffuse_survey_good_pointings.txt"
    version = 5
    subversion = 2
    quiet = False
    delete = False

    obsfile = open(obsfile_name, "r")
    obsids = [line.split( ) for line in obsfile.readlines()]
    obsids = [obs[0] for obs in obsids]
    obsfile.close()
    nonredundant_obsids = list(set(obsids))
    if len(obsids) != len(nonredundant_obsids):
        print "WARNING: Obs list contains redundant entries."
        obsids = nonredundant_obsids

    t = Time([int(obsid) for obsid in obsids], format="gps", scale="utc")
    jds = t.jd
    jds = [int(jd) for jd in jds]
    save_directories = ["EoRuvfits/jd" + str(jd) + "v"+ str(version) + "_" + str(subversion) + "/" for jd in jds]

    all_nodes = ["eor-02", "eor-03", "eor-04", "eor-05","eor-07", "eor-08", "eor-10", "eor-11", "eor-12", "eor-13", "eor-14"]
    all_nodes = ["/nfs/" + nodename + "/r1/" for nodename in all_nodes]

    not_found = 0
    for i, obsid in enumerate(obsids):
        uvfits_found = False
        metafits_found = False
        for node in all_nodes:
            if os.path.isdir(node + save_directories[i]):
                if len(os.listdir(node + save_directories[i])) == 0:
                    if delete:
                        print "   Directory " + node + save_directories[i] + " is empty -- deleting directory"
                        os.rmdir(node + save_directories[i])
                else:
                    if os.path.isdir(node + save_directories[i] + obsid + "/"):
                        if len(os.listdir(node + save_directories[i] + obsid + "/")) == 0:
                            if delete:
                                print "   Directory " + node + save_directories[i] + obsid + "/ is empty -- deleting directory"
                                os.rmdir(node + save_directories[i] + obsid + "/")
                        else:
                            directory_contents = os.listdir(node + save_directories[i] + obsid + "/")
                            if not quiet:
                                print "   Directory " + node + save_directories[i] + obsid + "/ contains the following files:"
                            for filename in directory_contents:
                                if not quiet:
                                    print "      " + filename
                                if filename == obsid + ".metafits":
                                    metafits_found = True
                                if filename == obsid + ".uvfits":
                                    uvfits_found = True
            if uvfits_found and metafits_found:
                print obsid + " located in " + node + save_directories[i]
                break
            else:
                uvfits_found = False
                metafits_found = False
        if not uvfits_found or not metafits_found:
            print obsid + "    DATA NOT FOUND."
            not_found += 1
    print str(not_found) + " obsids not found."


if __name__ == '__main__':
    main()
