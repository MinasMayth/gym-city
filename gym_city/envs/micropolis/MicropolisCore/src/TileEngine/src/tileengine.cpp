/* tileengine.cpp
 *
 * Micropolis, Unix Version.  This game was released for the Unix platform
 * in or about 1990 and has been modified for inclusion in the One Laptop
 * Per Child program.  Copyright (C) 1989 - 2007 Electronic Arts Inc.  If
 * you need assistance with this program, you may contact:
 *   http://wiki.laptop.org/go/Micropolis  or email  micropolis@laptop.org.
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.  You should have received a
 * copy of the GNU General Public License along with this program.  If
 * not, see <http://www.gnu.org/licenses/>.
 * 
 *             ADDITIONAL TERMS per GNU GPL Section 7
 * 
 * No trademark or publicity rights are granted.  This license does NOT
 * give you any right, title or interest in the trademark SimCity or any
 * other Electronic Arts trademark.  You may not distribute any
 * modification of this program using the trademark SimCity or claim any
 * affliation or association with Electronic Arts Inc. or its employees.
 * 
 * Any propagation or conveyance of this program must include this
 * copyright notice and these terms.
 * 
 * If you convey this program (or any modifications of it) and assume
 * contractual liability for the program to recipients of it, you agree
 * to indemnify Electronic Arts for any liability that those contractual
 * assumptions impose on Electronic Arts.
 * 
 * You may not misrepresent the origins of this program; modified
 * versions of the program must be marked as such and not identified as
 * the original program.
 * 
 * This disclaimer supplements the one included in the General Public
 * License.  TO THE FULLEST EXTENT PERMISSIBLE UNDER APPLICABLE LAW, THIS
 * PROGRAM IS PROVIDED TO YOU "AS IS," WITH ALL FAULTS, WITHOUT WARRANTY
 * OF ANY KIND, AND YOUR USE IS AT YOUR SOLE RISK.  THE ENTIRE RISK OF
 * SATISFACTORY QUALITY AND PERFORMANCE RESIDES WITH YOU.  ELECTRONIC ARTS
 * DISCLAIMS ANY AND ALL EXPRESS, IMPLIED OR STATUTORY WARRANTIES,
 * INCLUDING IMPLIED WARRANTIES OF MERCHANTABILITY, SATISFACTORY QUALITY,
 * FITNESS FOR A PARTICULAR PURPOSE, NONINFRINGEMENT OF THIRD PARTY
 * RIGHTS, AND WARRANTIES (IF ANY) ARISING FROM A COURSE OF DEALING,
 * USAGE, OR TRADE PRACTICE.  ELECTRONIC ARTS DOES NOT WARRANT AGAINST
 * INTERFERENCE WITH YOUR ENJOYMENT OF THE PROGRAM; THAT THE PROGRAM WILL
 * MEET YOUR REQUIREMENTS; THAT OPERATION OF THE PROGRAM WILL BE
 * UNINTERRUPTED OR ERROR-FREE, OR THAT THE PROGRAM WILL BE COMPATIBLE
 * WITH THIRD PARTY SOFTWARE OR THAT ANY ERRORS IN THE PROGRAM WILL BE
 * CORRECTED.  NO ORAL OR WRITTEN ADVICE PROVIDED BY ELECTRONIC ARTS OR
 * ANY AUTHORIZED REPRESENTATIVE SHALL CREATE A WARRANTY.  SOME
 * JURISDICTIONS DO NOT ALLOW THE EXCLUSION OF OR LIMITATIONS ON IMPLIED
 * WARRANTIES OR THE LIMITATIONS ON THE APPLICABLE STATUTORY RIGHTS OF A
 * CONSUMER, SO SOME OR ALL OF THE ABOVE EXCLUSIONS AND LIMITATIONS MAY
 * NOT APPLY TO YOU.
 */

/** @file tileengine.cpp */

////////////////////////////////////////////////////////////////////////
// Includes

#include <iostream>
#include "stdafx.h"


/** @todo Make an interface class, and move the global below to a derived class */


////////////////////////////////////////////////////////////////////////
// Globals


// Interface to PyCairo functions.
Pycairo_CAPI_t *Pycairo_CAPI;

////////////////////////////////////////////////////////////////////////
// TileEngine class

TileEngine::TileEngine() :
    width(0),
    height(0),
    bufData(NULL),
    colBytes(0),
    rowBytes(0),
    tileFormat(TILE_FORMAT_BYTE_UNSIGNED),
    floatOffset((float)0.0),
    floatScale((float)1.0),
    tileShift(0),
    tileMask(~0)
{
}


TileEngine::~TileEngine()
{
}


void TileEngine::setBuffer(
   void *buf)
{
    bufData = buf;
}


unsigned long TileEngine::getValue(
    int col,
    int row,
    PyObject *tileFunction,
    const int *tileMapData,
    unsigned int tileMapCount)
{
    if ((bufData == NULL) ||
	(col < 0) ||
	(col >= width) ||
	(row < 0) ||
	(row >= height)) {
	return 0;
    }

    int tile = 0;

    switch (tileFormat) {

	case TILE_FORMAT_BYTE_UNSIGNED:
	    tile = 
		(int)(
		    *(unsigned char *)(
			(unsigned char *)bufData +
			(col * colBytes) +
			(row * rowBytes)));
	    break;

	case TILE_FORMAT_BYTE_SIGNED:
	    tile = 
		(int)(
		    *(char *)(
			(char *)bufData +
			(col * colBytes) +
			(row * rowBytes)));
	    break;

	case TILE_FORMAT_SHORT_UNSIGNED:
	    tile = 
		(int)(
		    *(unsigned short *)(
			(unsigned char *)bufData +
			(col * colBytes) +
			(row * rowBytes)));
	    break;

	case TILE_FORMAT_SHORT_SIGNED:
	    tile = 
		(int)(
		    *(short *)(
			(unsigned char *)bufData +
			(col * colBytes) +
			(row * rowBytes)));
	    break;

	case TILE_FORMAT_LONG_UNSIGNED:
	    tile =
		(int)(
		    0x7fffffff &
		    *(unsigned long *)(
			(unsigned char *)bufData +
			(col * colBytes) +
			(row * rowBytes)));
	    break;

	case TILE_FORMAT_LONG_SIGNED:
	    tile =
		(int)(
		    *(unsigned long *)(
			(unsigned char *)bufData +
			(col * colBytes) +
			(row * rowBytes)));
	    break;

	case TILE_FORMAT_FLOAT:
	    tile =
		(int)(
		    floatOffset +
		    (floatScale *
			*(float *)(
			    (unsigned char *)bufData +
			    (col * colBytes) +
			    (row * rowBytes))));
	      break;

	case TILE_FORMAT_DOUBLE:
	    tile =
		(int)(
		    floatOffset +
		    (floatScale *
			*(double *)(
			    (unsigned char *)bufData +
			    (col * colBytes) +
			    (row * rowBytes))));
	    break;

	default:
	    tile = 0;
	    break;

    }

    if (tileFunction &&
	tileFunction != Py_None) {
	// Call tile function.
	PyObject *result =
	    PyObject_CallFunction(
		tileFunction,
		"iii",
		col,
		row,
		tile);
	if (!PyNumber_Check(result)) {
	      PyErr_SetString(
		  PyExc_TypeError,
		  "expected tileFunction to return an integer");
	} else {
	    tile = (unsigned long)PyLong_AsLong(result);
	}
	Py_DECREF(result);
    }

    tile <<= tileShift;
    tile &= tileMask;

    if (tileMapData != NULL) {
	tile = tileMapData[tile % tileMapCount];
    }

    return tile;
}


// Render fixed sized tiles, pre-drawn into the Cairo surface "tiles".
void TileEngine::renderTiles(
    cairo_t *ctx,
    cairo_surface_t *tilesSurf,
    int tilesWidth,
    int tilesHeight,
    PyObject *tileFunction,
    PyObject *tileMap,
    int tileSize,
    int renderCol,
    int renderRow,
    int renderCols,
    int renderRows,
    double alpha)
{

    if ((tileFunction != Py_None) &&
	!PyCallable_Check(tileFunction)) {
	PyErr_SetString(
	    PyExc_TypeError,
	    "expected tileFunction to be a callable function or None");
	return;
    }

    // The tileMap should be an array of 4 byte integers,
    // mapping virtual tiles indices to absolute tile numbers.
    if ((tileMap != Py_None) &&
	!PySequence_Check(tileMap)) {
	PyErr_SetString(
	    PyExc_TypeError,
	    "expected tileMap to be an array "
	    "of 4 byte integers or None");
	return;
    }

    const int *tileMapData = NULL;
    unsigned int tileMapCount = 0;

    if (tileMap != Py_None) {
	tileMapCount = (unsigned int)PySequence_Size(tileMap);
	Py_ssize_t tileMapLength = 0;
	if (PyObject_AsReadBuffer(
	    tileMap,
	    (const void **)&tileMapData,
	    &tileMapLength) != 0) {
	    PyErr_SetString(
		PyExc_TypeError,
		"expected tileMap with read buffer");
	    return;
	}

	int tileMapDataCount = 
	    (int)tileMapLength / sizeof(unsigned int);

	if (tileMapDataCount != (int)tileMapCount) {
	    PyErr_SetString(
		PyExc_TypeError,
		"expected tileMap read buffer of 4 byte integers");
	    return;
	}
    }

    int tilesPerRow = tilesWidth / tileSize;

    int renderX = renderCol * tileSize;
    int renderY = renderRow * tileSize;

    int r;
    for (r = 0; r < renderRows; r++) {
	int c;
	for (c = 0; c  < renderCols; c++) {
	    int col = (renderCol + c) % width;
	    int row = (renderRow + r) % height;

	    unsigned long tile =
		getValue(
		    col,
		    row,
		    tileFunction,
		    tileMapData,
		    tileMapCount);

	    double x = col * tileSize;
	    double y = row * tileSize;

	    // Tiles are arranged in a regular grid. 
	    // Calculate the position of the file in the 
	    // source tileSurf. 
	    int tileCol = tile % tilesPerRow;
	    int tileRow = tile / tilesPerRow;
	    int tileX = tileCol * tileSize;
	    int tileY = tileRow * tileSize;

	    // Draw a tile.

	    cairo_save(ctx);

	    cairo_translate(
		ctx,
		x - renderX,
		y - renderY);

	    cairo_rectangle(
		ctx,
		0,
		0,
		tileSize,
		tileSize);

	    cairo_clip(ctx);

	    cairo_set_source_surface(
		ctx,
		tilesSurf,
		-tileX,
		-tileY);

	    if (alpha >= 1.0) {
		cairo_paint(ctx);
	    } else {
		cairo_paint_with_alpha(ctx, alpha);
	    }

	    cairo_restore(ctx);

	}
    }
}


void TileEngine::renderTilesLazy(
    PyObject *pyctx,
    PyObject *tileFunction,
    PyObject *tileMap,
    int tileSize,
    int renderCol,
    int renderRow,
    int renderCols,
    int renderRows,
    double alpha,
    PyObject *tileGenerator,
    PyObject *tileCache,
    PyObject *tileCacheSurfaces,
    PyObject *tileState)
{
    cairo_t *ctx = PycairoContext_GET(pyctx);
    if ((tileFunction != Py_None) &&
        !PyCallable_Check(tileFunction)) {
        PyErr_SetString(
            PyExc_TypeError,
            "expected tileFunction to be a callable function or None");
        return;
    }
    
    // The tileMap should be None, or an array of 4 byte integers,
    // mapping virtual tiles indices to absolute tile numbers.

    const int *tileMapData = NULL;
    unsigned int tileMapCount = 0;

    if (tileMap != Py_None) {

      if (!PySequence_Check(tileMap)) {
            PyErr_SetString(
        	PyExc_TypeError,
        	"expected tileMap to be an array "
        	"of 4 byte integers or None");
            return;
        }

        tileMapCount = (unsigned int)PySequence_Size(tileMap);
        Py_ssize_t tileMapLength = 0;
        if (PyObject_AsReadBuffer(
            tileMap,
            (const void **)&tileMapData,
            &tileMapLength) != 0) {
            PyErr_SetString(
        	PyExc_TypeError,
        	"expected tileMap with read buffer");
            return;
        }

        int tileMapDataCount = 
            (int)tileMapLength / sizeof(unsigned int);

        if (tileMapDataCount != (int)tileMapCount) {
            PyErr_SetString(
        	PyExc_TypeError,
        	"expected tileMap read buffer of 4 byte integers");
            return;
        }
    }

    // The tileGenerator should be a function that takes one integer
    // tile parameter, and returns a tuple of three integers: a
    // surface index, a tileX and a tileY position.
    if (!PyCallable_Check(tileGenerator)) {
        PyErr_SetString(
            PyExc_TypeError,
            "expected tileGenerator callable object");
        return;
    }

    // The tileCache should be an array of integers, 4 integers per
    // tile.  The first is a "cached" flag, the second is a surface
    // index, the 3rd and 4th are a tileX and tileY position.

    int *tileCacheData = NULL;
    Py_ssize_t tileCacheLength = 0;

    if (PyObject_AsWriteBuffer(
	tileCache,
	(void **)&tileCacheData,
	&tileCacheLength) != 0) {
	PyErr_SetString(
	    PyExc_TypeError,
	    "expected tileCache array");
	return;
    }

    // The tileCacheSurfaces parameters should be a list of Cairo
    // surfaces.
    if (!PySequence_Check(tileCacheSurfaces)) {
	PyErr_SetString(
	    PyExc_TypeError,
	    "expected tileCacheSurfaces sequence");
	return;
    }

    // The tileState parameters should be None or a list of tile state
    // parameters.
    if ((tileState != Py_None) &&
	!PySequence_Check(tileState)) {
	PyErr_SetString(
	    PyExc_TypeError,
	    "expected tileState sequence or None");
	return;
    }

    int renderX = renderCol * tileSize;
    int renderY = renderRow * tileSize;

    int r;
    for (r = 0; r < renderRows; r++) {
	int c;
	for (c = 0; c  < renderCols; c++) {
	    int col = (renderCol + c) % width;
	    int row = (renderRow + r) % height;

	    unsigned long tile = 
		getValue(
		    col,
		    row,
		    tileFunction,
		    tileMapData,
		    tileMapCount);

	    double x = col * tileSize;
	    double y = row * tileSize;

	    // Get the tile surface index, tileX and tileY from the
	    // cache, or call the tileGenerator function to produce
	    // them, if they are not already cached.

	    int cacheOffset = tile * 4;
	    int tileSurfaceIndex = 0;
	    int tileX = 0;
	    int tileY = 0;

	    if (tileCacheData[cacheOffset + 0]) {

		// The tile is already cached. 
		// Get the values from the tileCache. 

		tileSurfaceIndex = tileCacheData[cacheOffset + 1];
		tileX = tileCacheData[cacheOffset + 2];
		tileY = tileCacheData[cacheOffset + 3];

	    } else {

		// The tile has not already been cached.  Call the
		// tileGenerator function to produce the tile, passing
		// it the absolute tile number as a parameter.

		// Mark the tile as cached, so we don't do this again. 
		tileCacheData[cacheOffset + 0] = 1;

		PyObject *result =
		    PyObject_CallFunction(
			tileGenerator,
			"i",
			tile);

		// The tile function returns a tuple of three numbers:
		// the surface index (into the tileCacheSurfaces array
		// of Cairo surfaces), the tileX and tileY position in
		// the surface.

		if (result == NULL) {
		    PyErr_SetString(
			PyExc_TypeError,
			"tile generator did not return a result");
		    return;
		}

		int success =
		    PyArg_ParseTuple(
			result,
			"iii",
			&tileSurfaceIndex,
			&tileX,
			&tileY);
		Py_DECREF(result);
		if (!success) {
		    PyErr_SetString(
			PyExc_TypeError,
			"tile generator return wrong number of "
		        "results in tuple");
		    return;
		}

		// Cache the returned values. 
		tileCacheData[cacheOffset + 1] = tileSurfaceIndex;
		tileCacheData[cacheOffset + 2] = tileX;
		tileCacheData[cacheOffset + 3] = tileY;
	    }

	    // Get the Python object wrapping the Cairo surface for
	    // the tile.
	    PyObject *tiles =
		PySequence_GetItem(
		    tileCacheSurfaces,
		    tileSurfaceIndex);

	    if (tiles == NULL) {
		PyErr_SetString(
		    PyExc_TypeError,
		    "tile generator returned invalid tile "
		    "surface index");
		return;
	    }

	    // TODO: this typecheck throws segfault

////////    if (!PyObject_TypeCheck(
////////	  tiles,
////////	  &PycairoSurface_Type)) {
////////	PyErr_SetString(
////////	    PyExc_TypeError,
////////	    "expected cairo_surface_t objects "
////////	    "in tileCacheSurfaces");
////////	return;
////////    }

	    // Get the cairo_surface_t from the Python object.
	    cairo_surface_t *tilesSurf = PycairoSurface_GET(tiles);
	    Py_DECREF(tiles);

	    // Draw a tile.

	    cairo_save(ctx);

	    cairo_translate(
		ctx,
		x - renderX,
		y - renderY);

	    cairo_rectangle(
		ctx,
		0,
		0,
		tileSize,
		tileSize);

	    cairo_clip(ctx);

	    cairo_set_source_surface(
		ctx,
		tilesSurf,
		-tileX,
		-tileY);

	    if (alpha >= 1.0) {
		cairo_paint(ctx);
	    } else {
		cairo_paint_with_alpha(ctx, alpha);
	    }

	    cairo_restore(ctx);
	}
    }
}
  

void TileEngine::renderPixels(
    cairo_surface_t *destSurf,
    cairo_surface_t *cmapSurf,
    PyObject *tileFunction,
    PyObject *tileMap,
    int renderCol,
    int renderRow,
    int renderCols,
    int renderRows)
{
    if ((tileFunction != Py_None) &&
	!PyCallable_Check(tileFunction)) {
	PyErr_SetString(
	    PyExc_TypeError,
	    "expected tileFunction to be a callable function or None");
	return;
    }

    // The tileMap should be None, or an array of 4 byte integers,
    // mapping virtual tiles indices to absolute tile numbers.

    const int *tileMapData = NULL;
    unsigned int tileMapCount = 0;

    if (tileMap != Py_None) {

        if (!PySequence_Check(tileMap)) {
	    PyErr_SetString(
		PyExc_TypeError,
		"expected tileMap to be an array "
		"of 4 byte integers or None");
	    return;
	}

	tileMapCount = (unsigned int)PySequence_Size(tileMap);
	Py_ssize_t tileMapLength = 0;
	if (PyObject_AsReadBuffer(
	    tileMap,
	    (const void **)&tileMapData,
	    &tileMapLength) != 0) {
	    PyErr_SetString(
		PyExc_TypeError,
		"expected tileMap with read buffer");
	    return;
	}

	int tileMapDataCount = 
	    (int)tileMapLength / sizeof(unsigned int);

	if (tileMapDataCount != (int)tileMapCount) {
	  PyErr_SetString(
	    PyExc_TypeError,
	    "expected tileMap read buffer of 4 byte integers");
	  return;
	}
    }

    unsigned char *destData = cairo_image_surface_get_data(destSurf);
    int destStride = cairo_image_surface_get_stride(destSurf);

    unsigned char *cmapData = cairo_image_surface_get_data(cmapSurf);
    int cmapWidth = cairo_image_surface_get_width(cmapSurf);
    int cmapStride = cairo_image_surface_get_stride(cmapSurf);

    int r;
    for (r = 0; r < renderRows; r++) {
	int c;
	for (c = 0; c  < renderCols; c++) {
	    int col = (renderCol + c) % width;
	    int row = (renderRow + r) % height;

	    unsigned long tile = 
		getValue(
		    col,
		    row,
		    tileFunction,
		    tileMapData,
		    tileMapCount);

	    int sourceX = tile % cmapWidth;
	    int sourceY = tile / cmapWidth;

	    unsigned char *sourcePixel =
		cmapData +
		(sourceX * 4) +
		(sourceY * cmapStride);

	    unsigned char *destPixel =
		destData +
		(c * 4) +
		(r * destStride);

	    *(long *)destPixel = *(long *)sourcePixel;
	}
    }
}


PyObject *TileEngine::getTileData(
    PyObject *tileFunction,
    PyObject *tileMap,
    int col,
    int row,
    int cols,
    int rows,
    int code,
    PyObject *tileViewCache)
{
    if ((tileFunction != Py_None) &&
	!PyCallable_Check(tileFunction)) {
	PyErr_SetString(
	    PyExc_TypeError,
	    "expected tileFunction to be a callable function or None");
	Py_INCREF(Py_None);
	return Py_None;
    }

    const int *tileMapData = NULL;
    unsigned int tileMapCount = 0;

    if (tileMap != Py_None) {

        if (!PySequence_Check(tileMap)) {
	    PyErr_SetString(
		PyExc_TypeError,
		"expected tileMap to be an array "
		"of 4 byte integers or None");
	    Py_INCREF(Py_None);
	    return Py_None;
	}

	tileMapCount = (unsigned int)PySequence_Size(tileMap);
	Py_ssize_t tileMapLength = 0;
	if (PyObject_AsReadBuffer(
	    tileMap,
	    (const void **)&tileMapData,
	    &tileMapLength) != 0) {
	    PyErr_SetString(
		PyExc_TypeError,
		"expected tileMap with read buffer");
	    Py_INCREF(Py_None);
	    return Py_None;
	}

	int tileMapDataCount = 
	    (int)tileMapLength / sizeof(unsigned int);

	if (tileMapDataCount != (int)tileMapCount) {
	    PyErr_SetString(
		PyExc_TypeError,
		"expected tileMap read buffer of 4 byte integers");
	    Py_INCREF(Py_None);
	    return Py_None;
	}
    }

    int *tileViewCacheData = NULL;
    int tileViewCacheCount = 0;

    if (tileViewCache != Py_None) {
	tileViewCacheCount =
	    (unsigned int)PySequence_Size(tileViewCache);
	Py_ssize_t tileViewCacheLength = 0;
	if ((tileViewCacheCount != (width * height)) ||
	    (PyObject_AsWriteBuffer(
	       tileViewCache,
	       (void **)&tileViewCacheData,
	       &tileViewCacheLength) != 0)) {
	    PyErr_SetString(
		PyExc_TypeError,
		"expected tileViewCache with write buffer");
	    Py_INCREF(Py_None);
	    return Py_None;
	}
    }

    const char *textCodeString =
	"0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ+-";
    int tileSize = sizeof(unsigned short);
    int r, c;
    int bufSize = 0;
    char *buf = NULL;
    bool returnBuffer = false;

    switch (code) {

	case TILE_CODE_RAW_BINARY_16: {
	    bufSize = tileSize * rows * cols;
	    buf = (char *)malloc(bufSize);
	    unsigned char *dst = (unsigned char *)buf;

	    for (r = 0; r < rows; r++) {
		for (c = 0; c < cols; c++) {
		    int tile = 
			getValue(
			    col + c, 
			    row + r, 
			    tileFunction, 
			    tileMapData, 
			    tileMapCount);
		    *dst++ = (tile >> 8) & 0xff; // High byte.
		    *dst++ = tile & 0xff; // Low byte.
		}
	    }
	    break;
	}

	case TILE_CODE_COMPRESSED_TEXT: {
	    bufSize = tileSize * rows * cols * 3; // to be safe
	    buf = (char *)malloc(bufSize);
	    unsigned char *dst = (unsigned char *)buf;
	    int tileIndex = 0;
	    int tileIndexMax = rows * cols;
	    int offset = 0;
	    int skip = 0;

	    while ((tileIndex < tileIndexMax) &&
		   (offset < bufSize)) {
		int c = tileIndex % cols;
		int r = tileIndex / cols;
		int tileCacheOffset = 
		    (col + c) + ((row + r) * width);
//		printf("tileIndex %d tileCacheOffset %d tileViewCacheCount %d c %d r %d cols %d rows %d ",
//		    tileIndex,
//		    tileCacheOffset,
//		    tileViewCacheCount,
//		    c, r, cols, rows);
		int tile = 
		    getValue(
			col + c, 
			row + r, 
			tileFunction,
			tileMapData, 
			tileMapCount);
//		printf("tile %d\n", tile);
		int curTile =
		    (tileViewCacheData == NULL)
			? -1
			: tileViewCacheData[tileCacheOffset];

		if (tile == curTile) {
		    skip++;
		} else {
//		    printf("tile %d skip %d offset %d r %d c %d index %d\n", tile, skip, offset, r, c, c + (r * width));
		    if (skip) {
			if (skip == 1) {
//			    printf("skip 1\n");
			    *dst++ = TILE_CODE_COMPRESSED_TEXT_SKIP_0;
			    offset++;
			} else if (skip == 2) {
//			    printf("skip 2/1\n");
			    *dst++ = TILE_CODE_COMPRESSED_TEXT_SKIP_0;
			    *dst++ = TILE_CODE_COMPRESSED_TEXT_SKIP_0;
			    offset += 2;
			} else if (skip < 64) {
//			    printf("skip %d/64\n", skip);
			    int val = skip;
			    *dst++ = TILE_CODE_COMPRESSED_TEXT_SKIP_1;
			    *dst++ = textCodeString[val];
			    offset += 2;
			} else if (skip < 4096) {
//			    printf("skip %d/4096\n", skip);
			    int val = skip;
			    *dst++ = TILE_CODE_COMPRESSED_TEXT_SKIP_2;
			    *dst++ = textCodeString[(val >> 6) & 63]; // High.
			    *dst++ = textCodeString[val & 63]; // Low.
			    offset += 3;
			} else {
//			    printf("skip %d/999999\n", skip);
			    int val = skip;
			    *dst++ = TILE_CODE_COMPRESSED_TEXT_SKIP_3;
			    *dst++ = textCodeString[(val >> 12) & 63]; // Highest.
			    *dst++ = textCodeString[(val >> 6) & 63]; // High.
			    *dst++ = textCodeString[val & 63]; // Low.
			    offset += 4;
			}

			skip = 0;

		    }

		    if (tileViewCacheData != NULL) {
			tileViewCacheData[tileCacheOffset] = tile;
		    }

		    int low = tile & 63;
		    int high = (tile >> 6) & 63;
		    *dst++ = textCodeString[low];
		    *dst++ = textCodeString[high];

		    offset += 2;
		}

		tileIndex++;
	    }

	    bufSize = offset;

	    break;
	}

	case TILE_CODE_COMPRESSED_BINARY_16: {
	    //returnBuffer = true; // @todo Fix pyamf code to convert buffer to ByteArray.
	    bufSize = tileSize * rows * cols * 3; // to be safe
	    buf = (char *)malloc(bufSize);
	    unsigned char *dst = (unsigned char *)buf;
	    int tileIndex = 0;
	    int tileIndexMax = rows * cols;
	    int offset = 0;
	    int skip = 0;

	    while ((tileIndex < tileIndexMax) &&
		   (offset < bufSize)) {
		int c = tileIndex % cols;
		int r = tileIndex / cols;
		int tileCacheOffset = 
		    (col + c) + ((row + r) * width);
//		printf("tileIndex %d tileCacheOffset %d tileViewCacheCount %d c %d r %d cols %d rows %d\n",
//		    tileIndex,
//		    tileCacheOffset,
//		    tileViewCacheCount,
//		    c, r, cols, rows);
		int tile = 
		    getValue(
			col + c, 
			row + r, 
			tileFunction,
			tileMapData, 
			tileMapCount);
//		printf("tile %d\n", tile);
		int curTile =
		    (tileViewCacheData == NULL)
			? -1
			: tileViewCacheData[tileCacheOffset];

		if (tile == curTile) {
		    skip++;
		} else {
//		    printf("tile %d skip %d offset %d r %d c %d index %d\n", tile, skip, offset, r, c, c + (r * width));
		    while (skip > 0) {
			if (skip == 1) {
//			    printf("skip 1/1\n");
			    // Skip of 1 is one single skip.
			    //printf("Skip of 1 is one single skip.\n");
			    *dst++ = TILE_CODE_COMPRESSED_BINARY_16_SKIP_0;
			    offset++;
			    skip = 0;
			} else if (skip == 2) {
//			    printf("skip 2/2\n");
			    // Skip of 2 is two single skips.
			    //printf("Skip of 2 is two single skips.\n");
			    *dst++ = TILE_CODE_COMPRESSED_BINARY_16_SKIP_0;
			    *dst++ = TILE_CODE_COMPRESSED_BINARY_16_SKIP_0;
			    offset += 2;
			    skip = 0;
			} else if (skip < (3 + 256)) {
//			    printf("skip %d/258 val %d\n", skip, skip - 3);
			    // Skip of 3 is val of 0.
			    // Skip of (3 + 255)=258 is val of 255.
			    // Skip range 3..258 for val range 0..255.
			    int val = skip - 3;
			    //printf("Skip %d range 3..258 for val %d range 0..255.\n", skip, val);
			    *dst++ = TILE_CODE_COMPRESSED_BINARY_16_SKIP_1;
			    *dst++ = val;
			    offset += 2;
			    skip = 0;
			} else {
//			    printf("skip %d/65794 val %d\n", skip, skip - (3 + 256));
			    // Skip of (3 + 256)=259 is val of 0.
			    // Skip of (3 + 256 + 65535)=65794 is val of 65535.
			    // Skip range 259..65794 for val range 0..65535.
			    int val = skip - (3 + 256);

			    // Skips greater than 65794 are broken up into repeated skips
			    // by the while loop above.
			    if (val > 65535) {
			        val = 65535;
			    }

			    //printf("Skip %d range 259..65794 for val %d range 0..65535.\n", skip, val);

			    int high = (val >> 8) & 0xff;
			    int low = val & 0xff;

			    *dst++ = TILE_CODE_COMPRESSED_BINARY_16_SKIP_2;
			    *dst++ = high;
			    *dst++ = low;

			    offset += 3;
			    skip -= val + (3 + 256);
			}
		    }

		    if (tileViewCacheData != NULL) {
			tileViewCacheData[tileCacheOffset] = tile;
		    }

		    int high = (tile >> 8) & 0xff;
		    int low = tile & 0xff;

		    assert(high <= TILE_CODE_COMPRESSED_BINARY_16_HIGH_MAX);

		    *dst++ = high;
		    *dst++ = low;
		    offset += 2;
		    //printf("tile high 0x%02X low 0x%02X\n", high, low);
		}

		tileIndex++;
	    }

	    bufSize = offset;

	    break;
	}

    }

    PyObject *result;

    if (returnBuffer) {
	result =
	    PyTuple_New(bufSize);
        unsigned char *destBuf = NULL;
        Py_ssize_t destSize = 0;
	PyObject_AsWriteBuffer(
	    result,
	    (void **)&destBuf,
	    &destSize);
	assert(destSize == bufSize);
	memcpy(
	    destBuf,
	    buf,
	    destSize);
    } else {
	result =
	    PyBytes_FromStringAndSize(
		(const char *)buf,
		bufSize);
    }

    free(buf);

    return result;
}


////////////////////////////////////////////////////////////////////////
