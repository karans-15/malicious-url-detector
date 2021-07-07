#!/bin/bash
 echo
 for domain in $@; do
 echo --------------------
 echo $domain
 echo --------------------
 curl -sILk $domain | egrep 'HTTP|Loc' | sed 's/Loc/ -> Loc/g'
 echo
 done
