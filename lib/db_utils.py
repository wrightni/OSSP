import os
import glob
import datetime
import sys
from osgeo import osr
import sqlite3
from utils import Task


def create_task_list_db(db_filepath):
    '''
    -----> Only selecting p002 for now, update this to process all parts!
    Creates a task list from the given database. Variables are specific
    to a custom implementation of a database and directory structure.
    '''
    base_dir = '/media/sequoia/DigitalGlobe/imagery'
    out_dir = '/media/sequoia/DigitalGlobe/processed'
    task_list = []

    # Open the database
    conn = sqlite3.connect(db_filepath)

    # Select the images that need to be processed with a database query
    cursor = conn.execute("SELECT NAME FROM DigitalGlobe WHERE CLOUD = 1 \
                                                            AND SENSOR = 'Pan_MS1_MS2' \
                                                            AND LOCAL = 1 \
                                                            AND QA IS NULL")
    for row in cursor:
        image_id = row[0]

        # Return a list of all MS .ntf files that match the image id
        # -----> Only selecting p002 for now, update this to process all parts!
        image_list = glob.glob('{}/*{}*M1BS*P002.ntf'.format(base_dir, image_id))

        for image in image_list:
            # Just in case there are any hidden files
            if image[0] == '.':
                continue

            image_name = os.path.split(image)[-1]
            # Add this image to the task list
            new_task = Task(image_name, base_dir)
            new_task.set_dst_dir(out_dir)

            task_list.append(new_task)

    # Close the database
    conn.close()

    return task_list


def redirect_output(folder):
    current_date = datetime.datetime.today().strftime('%Y-%m-%d')
    filename = os.path.join(folder, "{}.log".format(current_date))
    print("Redirecting output to: {}".format(filename))
    save_stdout = sys.stdout
    save_stderr = sys.stderr

    fh = open(filename, 'w')
    sys.stdout = fh
    sys.stderr = fh

    return fh, save_stdout, save_stderr


def write_to_database(db_filepath, image_id, part, pixel_counts,
                      vcode, vtds, gsd, cc):
    '''
    INPUT:
        db_name: filename of the database
        path: location where the database is stored
        pixel_clounts: number of pixels in each classification category
            [snow, gray, melt, water, shadow]

    Writes the classification pixel counts to the database at the image_id entry
    NOTES:
        For now, this overwrites existing data.
    FUTURE:
        Develop method that checks for existing data and appends the current
            data to that, record which parts contributed to the total.

    '''

    #part_num = os.path.splitext(image_name)[0].split('_')[-1]

    # Convert pixel_counts into percentages and total area
    area = 1  # Prevent division by 0
    for i in range(len(pixel_counts)):
        area +=  pixel_counts[i]
    prcnt = []
    for i in range(len(pixel_counts)):
        prcnt.append(float(pixel_counts[i] / area))

    # Convert area to square kilometers
    area_km = int(area * gsd / 1000000)

    current_date = datetime.datetime.today().strftime('%Y-%m-%d')

    # Open the database
    conn = sqlite3.connect(db_filepath)
    # Update the entry at image_id with the given pixel counts
    update_cmd = ("UPDATE DigitalGlobe " +
                  "SET AREA = {0:d}, SNOW = {1:f}, GRAY = {2:f}, MP = {3:f}, OW = {4:f}, " +
                  "UL = '({5:0.3f}, {6:0.3f})', LL = '({7:0.3f}, {8:0.3f})', " +
                  "UR = '({9:0.3f}, {10:0.3f})', LR = '({11:0.3f}, {12:0.3f})', " +
                  "PART = '{13:s}', PDATE = '{14:s}', VCODE = '{15:s}', VTDS = '{16:s}' " +
                  "WHERE NAME = '{17:s}'").format(area_km, prcnt[0], prcnt[1], prcnt[2], prcnt[3],
                                                 cc[0][0], cc[0][1], cc[1][0], cc[1][1],
                                                 cc[2][0], cc[2][1], cc[3][0], cc[3][1],
                                                 part, current_date, vcode, vtds,
                                                 image_id)
    print(update_cmd)
    # Commit the changes
    conn.commit()
    # Close the database
    conn.close()


def corner_coord_transform(src_dataset):
    '''
    Finds the resolution and corner coordinates of a gdal dataset
    :param src_dataset: gdal image dataset
    :return: resolution (gsd) and corner coordinates (geo_ext)
    '''
    gt = src_dataset.GetGeoTransform()
    cols = src_dataset.RasterXSize
    rows = src_dataset.RasterYSize
    ext = GetExtent(gt, cols, rows)
    gsd = (gt[1] - gt[5]) / 2.0

    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src_dataset.GetProjection())
    tgt_srs=osr.SpatialReference()
    tgt_srs.ImportFromEPSG(4326)
    # tgt_srs = src_srs.CloneGeogCS()

    geo_ext = ReprojectCoords(ext, src_srs, tgt_srs)

    return gsd, geo_ext


def GetExtent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
        yarr.reverse()
    return ext


def ReprojectCoords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords