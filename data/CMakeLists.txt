# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

project(data)

set(TEMPLATE_DIRNAME "templates")

set(TEMPLATE_DIRPATH ${CMAKE_CURRENT_SOURCE_DIR}/${TEMPLATE_DIRNAME})
file(GLOB_RECURSE CONFIG_FILES
  ${TEMPLATE_DIRPATH}/*
  )

foreach(CONFIG_FILE ${CONFIG_FILES})
  file(RELATIVE_PATH RELATIVE_FILEPATH ${TEMPLATE_DIRPATH} ${CONFIG_FILE})
  configure_file(${CONFIG_FILE} "${RELATIVE_FILEPATH}")
endforeach()

install(
        DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/Models
        ${CMAKE_CURRENT_BINARY_DIR}/Models
        ${CMAKE_CURRENT_BINARY_DIR}/Configuration
        DESTINATION "./"
)
