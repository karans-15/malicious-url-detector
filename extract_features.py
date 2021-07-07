#!/usr/bin/env python
# coding: utf-8

# # Extracting Features
# 
# Our Objective is to extract the features from the url so we can use these features further to make predictions and match our dataset

# In[1]:


#!c:/users/admin/appdata/local/programs/python/python39/python.exe -m pip install --upgrade pip
#!pip install favicon
#!pip install tldextract
#!pip install python-whois
#!pip install lxml
#!pip install tldextract

import os
import requests
from subprocess import *
from bs4 import BeautifulSoup
import json
import base64
from urllib.parse import urlparse
import favicon
import xml.etree.ElementTree as ET 
import tldextract
import datetime
from dateutil.relativedelta import relativedelta
import whois
import string


# In[2]:


from scipy.io import arff
import pandas as pd
import numpy as np


# In[3]:


# def getProcessedDataFrame(filepath):
#     dataset = arff.loadarff(filepath)
#     df = pd.DataFrame(dataset[0])
#     str_df = df.select_dtypes([np.object]) 
#     str_df = str_df.stack().str.decode('utf-8').unstack()

#     for col in str_df.columns:
#         str_df[col] = str_df[col].astype(int)
#     return str_df

# complete_training = getProcessedDataFrame("Training Dataset.arff")
# print(complete_training.columns)


# 1. Check if url has an IP address

# In[4]:


def has_ip_feature(url):
    
    #Use regEx to check for IP address 
    import re
    if re.match(r'http://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/.*', url) != None:
        return -1
    else:
        return 1


# 2. Check if url is long

# In[5]:


def long_url_feature(url):
    
    #Checks for long url
    if len(url)>75 :
        return -1
    elif len(url)>=54:
        return 0
    else:
        return 1


# 3. Check if url is shortened and get the complete ure

# In[6]:


def get_complete_url(shortened_url):
    
    #Returns expanded url if it is a shortened url
    command_stdout = Popen(['curl', shortened_url], stdout=PIPE).communicate()[0]
    #print(command_stdout)
    output = command_stdout.decode('utf-8')
    href_index = output.find("href=")
    if href_index == -1:
        href_index = output.find("HREF=")
    splitted_ = output[href_index:].split('"')
    expanded_url = splitted_[1]
    return expanded_url


def shortened_url_feature(url):
    famous_short_urls = ["bit.ly", "tinyurl.com", "goo.gl",
                       "rebrand.ly", "t.co", "youtu.be",
                       "ow.ly", "w.wiki", "is.gd"]

    domain = url.split("://")[1]
    #print(domain)
    domain = domain.split("/")[0]
    #print(domain)
    feature = 1
    if domain in famous_short_urls:
        feature = -1

    complete_url = None
    if feature == -1:
        complete_url = get_complete_url(url)

    return (feature, complete_url)


# 4. Checks if url has '@' in it

# In[7]:


def at_feature(url):
    feature = 1
    index = url.find("@")
    if index!=-1:
        feature = -1
  
    return feature


# 5. Check if url will redirect with '//'

# In[8]:


def redirect_feature(url):
    feature = 1
    if url.rindex("//")>7:
        feature = -1
    
    return feature


# 6. Check if url has a prefix '-'

# In[9]:


def prefix_feature(url):
    index = url.find("://")
    split_url = url[index+3:]
    #print(split_url)
    index = split_url.find("/")
    split_url = split_url[:index]
    #print(split_url)
    feature = 1
    index = split_url.find("-")
    #print(index)
    if index!=-1:
        feature = -1
  
    return feature


# 7. Check if url have sub domains/multi sub domains

# In[10]:


def multi_domains_feature(url):
    
    
    split_url = url.split("://")[1]
    split_url = split_url.split("/")[0]
    index = split_url.find("www.")
    if index!=-1:
        split_url = url[index+4:]
    # print(split_url)
    index = split_url.rfind(".")
    # print(index)
    if index!=-1:
        split_url = split_url[:index]
    # print(split_url)
    counter = 0
    for i in split_url:
        if i==".":
            counter+=1
  
    label = 1
    if counter==2:
        label = 0
    elif counter >=3:
        label = -1
  
    return label


# 8. Check if certificate issuer is trustworthy  (Gives phishing in most cases)

# In[11]:


# def authority_feature(url):
    
#     #Check for https
#     index_https = url.find("https://")
    
#     #Check for trustworthy issuer
#     valid_auth = ["GeoTrust", "GoDaddy", "Network Solutions", "Thawte", "Comodo", "Doster" , "VeriSign", "LinkedIn", "Sectigo",
#                 "Symantec", "DigiCert", "Network Solutions", "RapidSSLonline", "SSL.com", "Entrust Datacard", "Google", "Facebook"]
  
#     cmd = "curl -vvI " + url

#     stdout = Popen(cmd, shell=True, stderr=PIPE, env={}).stderr
#     output = stdout.read()
#     std_out = output.decode('UTF-8')
#     # print(std_out)
#     index = std_out.find("O=")

#     split = std_out[index+2:]
#     index_sp = split.find(" ")
#     cur = split[:index_sp]
  
#     index_sp = cur.find(",")
#     if index_sp!=-1:
#         cur = cur[:index_sp]
#     #print(cur)
#     label = -1
#     if cur in valid_auth and index_https!=-1:
#         label = 1
  
#     return label


# 9. Checking Domain registration length > 1 year  (Gives phishing always)

# In[12]:


# def domain_register_len_feature(u):
#     extract_res = tldextract.extract(u)
#     ul = extract_res.domain + "." + extract_res.suffix
#     try:
#         wres = whois.whois(u)
#         f = wres["Creation Date"][0]
#         s = wres["Registry Expiry Date"][0]
#         if(s>f+relativedelta(months=+12)):
#             return 1
#         else:
#             return -1
#     except:
#         return -1


# 10. Check if favicon loaded from external domain

# In[13]:


def favicon_feature(url):
    
    try:
        extract_res = tldextract.extract(url)
        url_ref = extract_res.domain

        favs = favicon.get(url)
        # print(favs)
        match = 0
        for favi in favs:
            url2 = favi.url
            extract_res = tldextract.extract(url2)
            url_ref2 = extract_res.domain

        if url_ref in url_ref2:
            match += 1

        if match >= len(favs)/2:
            return 1
        return -1
    except:
        return -1


# In[14]:


#11. Not using port feature


# 12. Check for https at the start

# In[15]:


def token_feature(u):
    # Assumption - pagename cannot start with this token
    ix = u.find("//https")
    if(ix==-1):
        return 1
    else:
        return -1


# 13. Check if less than 22% urls requested from same domain

# In[16]:


def request_url_feature(url):
    extract_res = tldextract.extract(url)
    url_ref = extract_res.domain

    command_stdout = Popen(['curl', 'https://api.hackertarget.com/pagelinks/?q=' + url], stdout=PIPE).communicate()[0]
    links = command_stdout.decode('utf-8').split("\n")

    count = 0

    for link in links:
        extract_res = tldextract.extract(link)
        url_ref2 = extract_res.domain

        if url_ref not in url_ref2:
            count += 1

    count /= len(links)

    if count < 0.22:
        return 1
    elif count < 0.61:
        return 0
    else:
        return -1


# 14. Check if anchor URLs less than 31%

# In[17]:


def url_validator(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

def url_anchor_feature(url):
    try:
        extract_res = tldextract.extract(url)
        url_ref = extract_res.domain
        html_content = requests.get(url).text
        soup = BeautifulSoup(html_content, "lxml")
        a_tags = soup.find_all('a')

        if len(a_tags) == 0:
            return 1

        invalid = ['#', '#content', '#skip', 'JavaScript::void(0)']
        bad_count = 0
        for t in a_tags:
            link = t["href"]

            if link in invalid:
                bad_count += 1

            if url_validator(link):
                extract_res = tldextract.extract(link)
                url_ref2 = extract_res.domain

                if url_ref not in url_ref2:
                    bad_count += 1

        bad_count /= len(a_tags)

        if bad_count < 0.31:
            return 1
        elif bad_count <= 0.67:
            return 0
        return -1
    except:
        return -1


# 15. Check if lot of links in <meta>, <script> and <link> tags

# In[18]:


def tags_feature(u):
    try:
        programhtml = requests.get(u).text
        s = BeautifulSoup(programhtml,"lxml")
        mtags = s.find_all('Meta')
        ud = tldextract.extract(u)
        upage = ud.domain
        mcount = 0
        for i in mtags:
            u1 = i['href']
            currpage = tldextract.extract(u1)
            u1page = currpage.domain
            if currpage not in ulpage:
                mcount+=1
        scount = 0
        stags = s.find_all('Script')
        for j in stags:
            u1 = j['href']
            currpage = tldextract.extract(u1)
            u1page = currpage.domain
            if currpage not in u1page:
                scount+=1
        lcount = 0
        ltags = s.find_all('Link')
        for k in ltags:
            u1 = k['href']
            currpage = tldextract.extract(u1)
            u1page = currpage.domain
            if currpage not in u1page:
                lcount+=1
        percmtag = 0
        percstag = 0
        percltag = 0

        if len(mtags) != 0:
            percmtag = (mcount*100)//len(mtags)
        if len(stags) != 0:
            percstag = (scount*100)//len(stags)
        if len(ltags) != 0:
            percltag = (lcount*100)//len(ltags)

        if(percmtag+percstag+percltag<17):
            return 1
        elif(percmtag+percstag+percltag<=81):
            return 0
        return -1
    except:
        return -1


# 16. Checks for SFH that contain empty string or about:blank

# In[19]:


def sfh_feature(u):
    try:
        programhtml = requests.get(u).text
        s = BeautifulSoup(programhtml,"lxml")
        try:
            f = str(s.form)
            ac = f.find("action")
            if(ac!=-1):
                i1 = f[ac:].find(">")
                u1 = f[ac+8:i1-1]
                if(u1=="" or u1=="about:blank"):
                    return -1
                er1 = tldextract.extract(u)
                upage = erl.domain
                erl2 = tldextract.extract(u1)
                usfh = erl2.domain
                if upage in usfh:
                    return 1
                return 0
            else:
                # Check this point
                return 1
        except:
            # Check this point
            return 1
    except:
        return -1


#  17. Check if information is being submitted to an email

# In[20]:


def submit_to_email_feature(url):
    try:
        html_content = requests.get(url).text
        soup = BeautifulSoup(html_content, "lxml")
        # Check if no form tag
        form_opt = str(soup.form)
        idx = form_opt.find("mail()")
        if idx == -1:
            idx = form_opt.find("mailto:")

        if idx == -1:
            return 1
        return -1
    except:
        return -1


# In[21]:


#18. Abnormal URL not including in features


# 19. Check if url is being redirected more than 1-2 times (Not used)

# In[22]:


# def forwarding_feature(url):
#     opt = Popen(["sh", "red.sh", url], stdout=PIPE).communicate()[0]
#     opt = opt.decode('utf-8')
#     # print(opt)
#     opt = opt.split("\n")
  
#     new = []
#     for i in opt:
#         i = i.replace("\r", " ")
#         new.extend(i.split(" "))
  

#     count = 0
#     for i in new:
   
#         if i.isdigit():
#             conv = int(i)
#             if conv > 300 and conv<310:
#                 count += 1

#     last_url = None
#     for i in new[::-1]:
#         if url_validator(i):
#             last_url = i
#             break

#     if (count<=1):
#         return 1, last_url
#     elif count>=2 and count <4:
#         return 0, last_url
#     return -1, last_url


# 20. Check if onmouseover changes url

# In[23]:


def onmouseover_feature(url):
    try:
        html_content = requests.get(url).text
    except:
        return -1
    soup = BeautifulSoup(html_content, "lxml")
    if str(soup).lower().find('onmouseover="window.status') != -1:
        return -1
    return 1


# 21. Check if rightclick is disabled

# In[24]:


def rightclick_feature(url):
    try:
        html_content = requests.get(url).text
        soup = BeautifulSoup(html_content, "lxml")
        if str(soup).lower().find("preventdefault()") != -1:
            return -1
        elif str(soup).lower().find("event.button==2") != -1:
            return -1
        elif str(soup).lower().find("event.button == 2") != -1:
            return -1
        return 1
    except:
        return -1


# In[25]:


#22. Pop up window feature wont be used


# 23. Check for frame border html

# In[26]:


def iframe_feature(url):
    try:
        html_content = requests.get(url).text
        soup = BeautifulSoup(html_content, "lxml")
        if str(soup.iframe).lower().find("frameborder") == -1:
            return 1
        return -1
    except:
        return -1


# 24. Check if age of domain is less than 6 months

# In[27]:


def age_of_domain_feature(url):
    try:
        extract_res = tldextract.extract(url)
        url_ref = extract_res.domain + "." + extract_res.suffix
        try:
            whois_res = whois.whois(url)
            if datetime.datetime.now() > whois_res["creation_date"][0] + relativedelta(months=+6):
                return 1
            else:
                return -1
        except:
            return -1
    except:
        return -1


# 25. Check if DNS Record exists

# In[28]:


def dns_record_feature(url):
    try:
        extract_res = tldextract.extract(url)
        url_ref = extract_res.domain + "." + extract_res.suffix
        try:
            whois_res = whois.whois(url)
            return 1
        except:
            return -1
    except:
        return -1


# 26. Check if website traffic < 100,000

# In[29]:


def web_traffic_feature(url):
    try:
        extract_res = tldextract.extract(url)
        url_ref = extract_res.domain + "." + extract_res.suffix
        html_content = requests.get("https://www.alexa.com/siteinfo/" + url_ref).text
        soup = BeautifulSoup(html_content, "lxml")
        value = str(soup.find('div', {'class': "rankmini-rank"}))[42:].split("\n")[0].replace(",", "")

        if not value.isdigit():
            return -1

        value = int(value)
        if value < 100000:
            return 1
        return 0
    except:
        return -1


# 27. Check if page rank < 0.2 

# In[30]:


def page_rank_feature(url):
    try:
        pageRankApi = open('pagerankAPI.txt').readline()
        extract_res = tldextract.extract(url)
        url_ref = extract_res.domain + "." + extract_res.suffix
        headers = {'API-OPR': pageRankApi}
        domain = url_ref
        req_url = 'https://openpagerank.com/api/v1.0/getPageRank?domains%5B0%5D=' + domain
        request = requests.get(req_url, headers=headers)
        result = request.json()
        # print(result)
        value = result['response'][0]['page_rank_decimal']
        if type(value) == str:
            value = 0

        if value < 2:
            return -1
        return 1
    except:
        return -1


# In[31]:


#28. Google Index not used 
#29. Links pointing to page not used


# 30. Check if host belongs to top phishing site or not (Not done cause it will need api keys and registration was shut)

# In[32]:


# def statistical_report_feature(url):
#     phishTankKey = open('/phishTankKey.txt')
#     phishTankKey = phishTankKey.readline()[:-1]

#     headers = {
#         'format': 'json',
#         'app_key': phishTankKey,
#         }

#     def get_url_with_ip(URI):
#         """Returns url with added URI for request"""
#         url = "http://checkurl.phishtank.com/checkurl/"
#         new_check_bytes = URI.encode()
#         base64_bytes = base64.b64encode(new_check_bytes)
#         base64_new_check = base64_bytes.decode('ascii')
#         url += base64_new_check
#         return url

#     def send_the_request_to_phish_tank(url, headers):
#         """This function sends a request."""
#         response = requests.request("POST", url=url, headers=headers)
#         return response

#     url = get_url_with_ip(url)
#     r = send_the_request_to_phish_tank(url, headers)

#     def parseXML(xmlfile): 

#         root = ET.fromstring(xmlfile) 
#         verified = False
#         for item in root.iter('verified'): 
#             if item.text == "true":
#                 verified = True
#                 break

#         phishing = False
#         if verified:
#             for item in root.iter('valid'): 
#                 if item.text == "true":
#                     phishing = True
#                     break

#         return phishing

#     inphTank = parseXML(r.text)
#     # print(r.text)

#     if inphTank:
#         return -1
#     return 1


# In[ ]:





# In[33]:


url = "https://tests.mettl.com/test-window-pi?key=GmIzQf0X8KA9tG7fOidIkEbMbqKup9l1Uez7WVPNnfqvFRcQQoVArm5BHZ1hyaVF"
url = "https://furniture-shop-vanilla-js.netlify.app/"

# print(has_ip_feature(url))
# print(long_url_feature(url))
# print(shortened_url_feature(url))
# print(at_feature(url))
# print(redirect_feature(url))
# print(prefix_feature(url))
# print(multi_domains_feature(url))
# print(authority_feature("https://www.amazon.com"))
# print(domain_register_len_feature("https://www.amazon.com"))  
# print(favicon_feature(url))
# print(token_feature(url))
# print(request_url_feature(url))
# print(url_anchor_feature(url))
# print(tags_feature(url))
# print(sfh_feature(url))
# print(submit_to_email_feature(url))
# print(forwarding_feature("https://www.amazon.com"))
# print(onmouseover_feature(url))
# print(rightclick_feature(url))
# print(iframe_feature(url))
# print(age_of_domain_feature(url))
# print(dns_record_feature(url))
# print(web_traffic_feature(url))
# print(page_rank_feature(url))


# In[34]:


def get_features(url):
    features_extracted = [0]*21
    phStatus, expanded = shortened_url_feature(url)

    if expanded is not None:
        if len(expanded) >= len(url):
            url = expanded

    print(url)
    features_extracted[0] = has_ip_feature(url)
    features_extracted[1] = long_url_feature(url)
    features_extracted[2] = phStatus
    features_extracted[3] = at_feature(url)
    features_extracted[4] = redirect_feature(url)
    features_extracted[5] = prefix_feature(url)
    features_extracted[6] = multi_domains_feature(url)
    features_extracted[7] = favicon_feature(url)
    features_extracted[8] = token_feature(url)
    features_extracted[9] = request_url_feature(url)
    features_extracted[10] = url_anchor_feature(url)
    features_extracted[11] = tags_feature(url)
    features_extracted[12] = sfh_feature(url)
    features_extracted[13] = submit_to_email_feature(url)
    features_extracted[14] = onmouseover_feature(url)
    features_extracted[16] = rightclick_feature(url)
    features_extracted[15] = iframe_feature(url)
    features_extracted[17] = age_of_domain_feature(url)
    features_extracted[18] = dns_record_feature(url)
    features_extracted[19] = web_traffic_feature(url)
    features_extracted[20] = page_rank_feature(url)
    
    return features_extracted


# In[35]:


#print(complete_training.columns)
#7,8,10,17,18,21,27,28,29
#SSLfinal_State,Domain_registration_length,port,Abnormal_URL,Redirect,popUpWindow,Google_Index,Links_pointing_to_page,Statistical_report


# In[36]:


# def convertEncodingToPositive(data):
#     mapping = {-1: 2, 0: 0, 1: 1}
#     i = 0
#     for col in data:
#         data[i] = mapping[col]
#         i+=1
#     return data


# In[37]:


# url = "https://www.github.com/karans-15"
# url = "https://furniture-shop-vanilla-js.netlify.app/"
# features = get_features(url)
# #print(features)
# features_extracted = convertEncodingToPositive(features)
# #print(features_extracted)


# In[38]:


# import numpy as np
# import pickle
# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder(sparse=False)


# # In[39]:


# one_hot_enc = pickle.load(open("One_Hot_Encoder", "rb"))
# transformed_point = one_hot_enc.transform(np.array(features_extracted).reshape(1, -1))

# model = pickle.load(open("RF_Final_Model.pkl", "rb"))
# prediction = model.predict(transformed_point)[0]

# if(prediction==1):
#     print("Website is SAFE!")
# elif(prediction==2):
#     print("DANGER!! This appears to be a phishing website")
# else:
#     print("Proceed with CAUTION, this seems Suspicious")

