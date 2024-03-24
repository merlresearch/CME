// Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#if _WIN32 || _WIN64
#define WIN32_LEAN_AND_MEAN
#include "targetver.h"
#endif

#include <stdio.h>
#if _WIN32 || _WIN64
#define WIN32_LEAN_AND_MEAN
#include <tchar.h>
#endif
