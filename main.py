from app import * 
from speech import *

def main():
    path_hugo = "dataset/hugo/"
    path_alex = "dataset/alex/"
    app = Application()
    app.load(path_hugo)
    app.load(path_alex)
    app.run()
    app.exitSafely()    

main()    
