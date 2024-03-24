// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "WallRemover.h"

#include <map>
#include <vector>
#include <string>
# include <iostream>
# include <random>

#include "HelperFunctions.h"


mjModel* warningSuppressLoadXML (const char* path_xml){
    char error[1000] = "Could not load XML file";

    FILE *fp;
    fp = fopen("/tmp/sample.txt", "w");
    _IO_FILE* save_stderr = stderr;

    stderr = fp;

    // suppress Qhull error message
    mjModel* model = mj_loadXML(path_xml, 0, error, 1000);

    stderr = save_stderr;
    fclose(fp);

    if( !model )
    {
        printf("%s\n", error);
    }

    return model;
}

WallConfig WallConfig::makeWallConfig(const std::string &maze_walls_file) {
    using namespace std;

    std::map<std::string, std::string> walls_config_map;

    if (!HelperFunctions::readConfig(maze_walls_file.c_str(), walls_config_map)) {
        throw std::runtime_error("could not read maze wall config.");
    }

    WallConfig wallConfig;
    std::string wallNames[] = {string("ring0"), string("ring1"), string("ring2"), string("ring3"), string("ring4")};

    for (int i = 0; i < 5; i++) {
        string curName = wallNames[i];

        if (walls_config_map[curName] == "yes") {
            wallConfig.useWalls[i] = true;
        } else if (walls_config_map[curName] == "no") {
            wallConfig.useWalls[i] = false;
        } else {
            throw std::runtime_error("Wall Config has invalid value.");
        }
    }

    return wallConfig;
}


// check if user specified a maze wall configuration file
// if so, then select only the walls according to the configuration file
// otherwise load the full model
mjModel* loadXMLWithWallLimitation(const char *filename, const WallConfig* wallConfig)
{
    std::vector<std::string> search_strings;

    if (wallConfig != NULL) {
        if (!wallConfig->useWalls[0]) {
            search_strings.push_back (std::string ("GameWallMesh0"));
        }
        if (!wallConfig->useWalls[1]) {
            search_strings.push_back (std::string ("GameWallMesh10"));
            search_strings.push_back (std::string ("GameWallMesh13"));
            search_strings.push_back (std::string ("GameWallMesh11"));
            search_strings.push_back (std::string ("GameWallMesh12"));
        }
        if (!wallConfig->useWalls[2]) {
            search_strings.push_back (std::string ("GameWallMesh20"));
            search_strings.push_back (std::string ("GameWallMesh21"));
            search_strings.push_back (std::string ("GameWallMesh22"));
            search_strings.push_back (std::string ("GameWallMesh23"));
        }
        if (!wallConfig->useWalls[3]) {
            search_strings.push_back (std::string ("GameWallMesh30"));
            search_strings.push_back (std::string ("GameWallMesh31"));
            search_strings.push_back (std::string ("GameWallMesh32"));
            search_strings.push_back (std::string ("GameWallMesh33"));
        }
        if (!wallConfig->useWalls[4]) {
            search_strings.push_back (std::string ("GameWallMesh4"));
        }
    }

    // now that we've made a list of geometry to discard, read the XML file, and then
    // discard whatever is in the list
    std::string xml_file;
    xml_file.assign (filename);

    std::ifstream is_file;
    is_file.open(xml_file);

    std::string deferred_line;

    // std::cout << xml_file << std::endl;

    std::string xml_tmp_file;

    if (is_file.is_open())
    {
        // we have a valid XML file open, let's write the new one which removes
        // unwanted geometry

        int len_str = xml_file.length ();
        std::string fname_no_ext = xml_file.substr (0, len_str - 4);

        // grab the milliseconds
        std::chrono::microseconds us = std::chrono::duration_cast< std::chrono::microseconds >(
                std::chrono::system_clock::now().time_since_epoch()
        );

        std::ostringstream oss;
        oss << us.count ();

        std::random_device rd;
        std::mt19937 mt(rd());
        int rand_no = mt();

        xml_tmp_file = fname_no_ext;
        xml_tmp_file.append ("_");
        xml_tmp_file.append (oss.str ());
        xml_tmp_file.append ("_");
        xml_tmp_file.append (std::to_string(rand_no));
        xml_tmp_file.append (".xml");

        // std::cout << xml_tmp_file << std::endl;

        std::ofstream os_file;
        os_file.open (xml_tmp_file, std::ofstream::out);

        if (os_file.is_open())
        {
            bool start_of_new_asset = false;

            std::string line;
            while( std::getline(is_file, line) )
            {
                // std::cout << line << std::endl;
                // std::istringstream is_line(line);

                if (start_of_new_asset)
                {
                    if (line.find ("asset") != std::string::npos)
                    {
                        if (line.find ("</") != std::string::npos)
                        {
                            //std::cout << "Found asset-end tag" << std::endl;

                            start_of_new_asset = false;
                        }
                        else
                        {
                            //std::cout << "Something seems wrong...." << std::endl;
                        }
                    }
                    else
                    {
                        // find if geometry in discard list
                        std::vector<std::string>::iterator	dl_iter = search_strings.begin ();

                        while (dl_iter != search_strings.end ())
                        {
                            if (line.find(*dl_iter) != std::string::npos)
                            {
                                break;
                            }

                            dl_iter++;
                        }
                        if (dl_iter == search_strings.end ())
                        {
                            //std::cout << "Not discarding anything for this asset."<< std::endl;

                            //std::cout << deferred_line << std::endl;
                            //std::cout << line << std::endl;

                            // we did not discard, write deferred line and this line
                            os_file << deferred_line << std::endl;
                            os_file << line << std::endl;

                            start_of_new_asset = false;
                        }
                        else
                        {
                            // we need to discard, skip writing to output until end of asset
                            //std::cout << "Discarding " << *dl_iter << std::endl;
                        }
                    }
                }
                else
                {
                    if (line.find ("asset") != std::string::npos)
                    {
                        if (line.find ("</") != std::string::npos)
                        {
                            //std::cout << "Found asset end tag" << std::endl;

                            os_file << line << std::endl;
                        }
                        else
                        {
                            // defer writing, since we may discard altogether
                            //std::cout << "Found asset tag" << std::endl;

                            start_of_new_asset = true;

                            deferred_line = line;
                        }
                    }
                    else if (line.find ("<geom") != std::string::npos)
                    {
                        //std::cout << "Found geom tag" << std::endl;

                        // find if geometry in discard list
                        std::vector<std::string>::iterator	dl_iter = search_strings.begin ();

                        while (dl_iter != search_strings.end ())
                        {
                            if (line.find(*dl_iter) != std::string::npos)
                            {
                                //std::cout << "Discarding: " << line << std::endl;
                                break;
                            }

                            dl_iter++;
                        }
                        if (dl_iter == search_strings.end ())
                        {
                            os_file << line << std::endl;
                        }
                    }
                    else
                        os_file << line << std::endl;
                }
            }
            os_file.close ();
        }
        is_file.close();
    }
    else
        std::cout << "Unable to open XML file " << filename << std::endl;

    mjModel* model = NULL;

    if (xml_tmp_file.length () != 0)
    {
        model = warningSuppressLoadXML(xml_tmp_file.c_str());
        std::remove (xml_tmp_file.c_str ());
    }

    return model;
}
