# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "C:/Users/u408220/esp/v5.0/esp-idf/components/bootloader/subproject"
  "C:/Users/u408220/esp/v5.0/esp-who/components/esp-dl/tutorial/df2img/build/bootloader"
  "C:/Users/u408220/esp/v5.0/esp-who/components/esp-dl/tutorial/df2img/build/bootloader-prefix"
  "C:/Users/u408220/esp/v5.0/esp-who/components/esp-dl/tutorial/df2img/build/bootloader-prefix/tmp"
  "C:/Users/u408220/esp/v5.0/esp-who/components/esp-dl/tutorial/df2img/build/bootloader-prefix/src/bootloader-stamp"
  "C:/Users/u408220/esp/v5.0/esp-who/components/esp-dl/tutorial/df2img/build/bootloader-prefix/src"
  "C:/Users/u408220/esp/v5.0/esp-who/components/esp-dl/tutorial/df2img/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/u408220/esp/v5.0/esp-who/components/esp-dl/tutorial/df2img/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/u408220/esp/v5.0/esp-who/components/esp-dl/tutorial/df2img/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
