
import os
import subprocess

def raw2orthorectify(filename, output_dir, script_path):
    #input_dir,input_file = os.path.split(filename)
    #ortho_dir1 = 'C:\Users\F003P1J\Desktop\DartmouthResearch\Image_Processing\OSSP\orthorectifiedPanImages'
    ## Orthorectify panchromatic image
    cmd = 'python {} --epsg 3413 --stretch ns --outtype UInt16 \
            {} {}'.format(script_path, filename, output_dir)

    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    proc.wait()

    ## Uncomment to output result
    output = proc.stdout.read()
    print output

def raw2pansharpened(pan_filename, ms_filename, output_dir, orth_path, pan_path):

    # Orthorecify the image
    #input_dir,input_file = os.path.split(pan_filename)
    #ortho_dir1 = 'C:\Users\F003P1J\Desktop\DartmouthResearch\Image_Processing\OSSP\orthorectifiedPanImages'
    ## Orthorectify panchromatic image
    cmd = 'python {} --epsg 3413 --stretch ns --outtype UInt16 \
            {} {}'.format(orth_path, pan_filename, output_dir)
    print cmd
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    proc.wait()

    ## Uncomment to output result
    output = proc.stdout.read()
    print output

    # Orthorectify multispectral image
    #input_dir,input_file = os.path.split(ms_filename)
    ortho_dir2 = 'C:\Users\F003P1J\Desktop\DartmouthResearch\Image_Processing\OSSP\orthorectifiedMSImages'
    ## Orthorectify panchromatic image
    cmd = 'python {} --epsg 3413 --stretch ns --outtype UInt16 \
            {} {}'.format(orth_path, pan_filename, output_dir)

    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    proc.wait()

    ## Uncomment to output result
    output = proc.stdout.read()
    print output

    # ortho_PanDir = os.listdir(output_dir)
    # ortho_MSDir = os.listdir(ortho_dir2)
    # Pansharpen the image
    pan_dir, pan_name = os.path.split(pan_filename)
    ms_dir, ms_name = os.path.split(ms_filename)
    for filename in os.listdir(output_dir):
        #print filename
        if filename == (pan_name[:-4] + '_u16ns3413.tif'):
            orthPanImage = filename
        #    print "orthpanimage is:" + orthPanImage
    #for filename in ortho_MSDir:
        elif filename == (ms_name[:-4] + '_u16ns3413.tif'):
            orthMSImage = filename
        #    print "orthmsimage is:" + orthMSImage
    #outputFile = orthMSImage[:-4]
    cmd = 'python  {} -of JP2OpenJPEG {}\{} {}\{} \
    {}_pansh.tif'.format(pan_path, output_dir, orthPanImage, output_dir, orthMSImage, orthMSImage[:-4])

    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    proc.wait()

    ## Uncomment to output result
    output = proc.stdout.read()
    print output

    #pgc_pansharpen?
    #gdal_pansharpen?


# panfile = 'C:\Users\F003P1J\Desktop\DartmouthResearch\Image_Processing\OSSP\June132014\WV02_20140613204443_10300100324B7D00_14JUN13204443-P1BS-500128648010_01_P002.ntf'
# msfile = 'C:\Users\F003P1J\Desktop\DartmouthResearch\Image_Processing\OSSP\June132014\WV02_20140613204443_10300100324B7D00_14JUN13204443-M1BS-500128648010_01_P002.ntf'
# raw2pansharpened(panfile, msfile)
