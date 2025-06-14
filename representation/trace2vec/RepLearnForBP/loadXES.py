from lxml import etree
import gensim
import os
import re
from nltk.tokenize import RegexpTokenizer
from gensim.models.doc2vec import TaggedDocument

def get_doc_list(folder_name):
    doc_list = []
    file_list = ['input/'+folder_name+'/'+name for name in os.listdir('input/'+folder_name) if name.endswith('xes')]

    print ('Found %s documents under the dir %s .....'%(len(file_list),folder_name))
    return file_list

def get_doc_XES_tagged(filename):
    tokenizer = RegexpTokenizer(r'\w+')
    taggeddoc = []
    tags=[]

    tree = etree.parse('input/'+filename)
    root= tree.getroot()
    index=0
    for element in root.iter():
        tag= element.tag.split('}')[1]
        if(tag== "trace"):
            wordslist = []
            tagslist = []
            for childelement in element.iterchildren():
                ctag= childelement.tag.split('}')[1]
                if(ctag== "string" and childelement.get('key')=='concept:name'):
                    doc_name=childelement.get('value')
                    #print(doc_name)
                elif (ctag== "event"):
                    for grandchildelement in childelement.iterchildren():
                        if(grandchildelement.get('key')=='concept:name'):
                            event_name=grandchildelement.get('value')
                        #    print(event_name)
                            wordslist.append(event_name.replace(' ',''))
            td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(wordslist))).split(),[index])
            taggeddoc.append(td)
            index= index +1
    return taggeddoc

def get_doc_XES_untagged(filename):
    tokenizer = RegexpTokenizer(r'\w+')
    docs = []
    texts = []

    tree = etree.parse('input/'+filename)
    root= tree.getroot()
    for element in root.iter():
        tag= element.tag.split('}')[1]
        if(tag== "trace"):
            # for tagged doc
            wordslist = []
            tagslist = []
            for childelement in element.iterchildren():
                ctag= childelement.tag.split('}')[1]
                if(ctag== "string" and childelement.get('key')=='concept:name'):
                    doc_name=childelement.get('value')
                elif (ctag=="event"):
                    for grandchildelement in childelement.iterchildren():
                        if(grandchildelement.get('key')=='concept:name'):
                            event_name=grandchildelement.get('value')
                            wordslist.append(event_name.replace(' ',''))
            texts.append(wordslist)
            td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(wordslist))).split(),[index])
            docs.append(td)
    return docs

def get_doc_multiple_XES_tagged(foldername):

    file_list= get_doc_list(foldername)
    taggeddoc = []
    traces = []
    tags=[]
    for filecounter,file in enumerate(file_list):
        tree = etree.parse(file)
        root= tree.getroot()
        index=0
        for element in root.iter():
            tag= element.tag.split('}')[1]
            if(tag== "trace"):
                wordslist = []
                for childelement in element.iterchildren():
                    ctag= childelement.tag.split('}')[1]
                    if(ctag== "string" and childelement.get('key')=='concept:name'):
                        doc_name=childelement.get('value')
                    elif (ctag== "event"):
                        for grandchildelement in childelement.iterchildren():
                            if(grandchildelement.get('key')=='concept:name'):
                                event_name=grandchildelement.get('value')
                                wordslist.append(event_name.replace(' ',''))
                if wordslist in traces:
                    td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(wordslist))).split(),[filecounter,tags[traces.index(wordslist)]])
                    taggeddoc.append(td)
                    #print('non-distinct trace found')
                else:
                    traces.append(wordslist)
                    tags.append(index)
                    td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(wordslist))).split(),[filecounter,index])
                    taggeddoc.append(td)
                    index= index +1

    return taggeddoc

def get_trace_names(filename):
    doc_names=[]
    tree = etree.parse('input/'+filename)
    root= tree.getroot()
    for element in root.iter():
        tag= element.tag.split('}')[1]
        if(tag== "trace"):
            for childelement in element.iterchildren():
                ctag= childelement.tag.split('}')[1]
                if(ctag== "string" and childelement.get('key')=='concept:name'):
                        doc_name=childelement.get('value')
                        doc_names.append(doc_name)
                        break
    return doc_names

def get_sentences_XES(filename):
    texts = []

    tree = etree.parse('input/'+filename)
    root= tree.getroot()
    for element in root.iter():
        tag= element.tag.split('}')[1]
        if(tag== "trace"):
            wordslist = []
            tagslist = []
            for childelement in element.iterchildren():
                ctag= childelement.tag.split('}')[1]
                if(ctag== "string" and childelement.get('key')=='concept:name'):
                    doc_name=childelement.get('value')
                elif (ctag=="event"):
                    for grandchildelement in childelement.iterchildren():
                        if(grandchildelement.get('key')=='concept:name'):
                            event_name=grandchildelement.get('value')
                        #    print(event_name)
                            wordslist.append(event_name.replace(' ',''))
            texts.append(wordslist)
    return texts
