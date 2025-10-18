
import * as dotenv from 'dotenv';
dotenv.config();

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';

import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';

import { Pinecone } from '@pinecone-database/pinecone';

import { PineconeStore } from '@langchain/pinecone';

async function indexDocument() {

    // pdf loading
    const PDF_PATH = './DSA_RAG.pdf';
    const pdfLoader = new PDFLoader(PDF_PATH);
    const rawDocs = await pdfLoader.load();
    console.log("PDF loaded");
    // console.log(rawDocs.length);

    //chuncking karo ab
    const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
    });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
    console.log("PDF chunked");





    //vector embedding k liye jao ab

    const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'text-embedding-004',
    });
    console.log("Embeddings created");

    // database me store karne k liye pinecone use karenge ab uska object banake

    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    console.log("Pinecone index connected");
    
    // langchain(chunking,embedding,db storage) ka use karenge ab

    
    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
    pineconeIndex,
    maxConcurrency: 5,
    });
    console.log("Document indexed successfully");






}

indexDocument();    