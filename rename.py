import glob, os

def rename(dir, pattern, titlePattern):
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        print("renaming" + title + " to " + titlePattern + title)
        os.rename(pathAndFilename, 
                  os.path.join(dir, titlePattern % title + ext))

rename(r'D:\MMP\TechnicalWork\wtcbkii', r'*.mid', r'wtcbkii%s')