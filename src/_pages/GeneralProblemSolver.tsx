// src/_pages/GeneralProblemSolver.tsx
import React, { useState, useEffect, useRef } from "react";
import { useQueryClient, useQuery } from "@tanstack/react-query";
import ScreenshotQueue from "../components/Queue/ScreenshotQueue";
import { useToast } from "../contexts/toast";
import { Screenshot } from "../types"; 
// --- START OF MODIFIED CODE ---
// import { marked } from 'marked'; // Default import can be problematic with .use() sometimes
import { Marked } from 'marked'; // Import the class
// --- END OF MODIFIED CODE ---
import katex from 'katex';
import 'katex/dist/katex.min.css'; 
import markedKatex from 'marked-katex-extension';

interface GeneralProblemSolverProps {
    currentLanguage: string;
    setLanguage: (language: string) => void;
    setView: (view: "queue" | "solutions" | "debug" | "generalProblem") => void;
}

// --- START OF MODIFIED CODE ---
// Configure a new Marked instance with the KaTeX extension
const katexExtensionOptions = {
  throwOnError: false, 
  delimiters: [
    { left: "$$", right: "$$", display: true },
    { left: "$", right: "$", display: false },
    { left: "\\(", right: "\\)", display: false },
    { left: "\\[", right: "\\]", display: true }
  ],
};

const customMarked = new Marked(); // Create an instance
customMarked.use(markedKatex(katexExtensionOptions)); // Apply extension to this instance
// --- END OF MODIFIED CODE ---


async function fetchGeneralScreenshots(): Promise<Screenshot[]> {
    try {
        const response = await window.electronAPI.getScreenshots();
        if (response && response.previews) {
            return response.previews.map((p: { path: string; preview: string; }) => ({
                id: p.path,
                path: p.path,
                timestamp: Date.now(),
                thumbnail: p.preview,
            }));
        }
        return [];
    } catch (error) {
        console.error("Failed to fetch screenshots for GeneralProblemSolver:", error);
        return [];
    }
}


const GeneralProblemSolver: React.FC<GeneralProblemSolverProps> = ({ setView }) => {
    const queryClient = useQueryClient();
    const { showToast } = useToast();
    const contentRef = useRef<HTMLDivElement>(null);
    const [isLoading, setIsLoading] = useState(false);
    
    const { data: solutionData, error: solutionError } = useQuery<{solution: string} | null>({
        queryKey: ["general_problem_solution"],
        queryFn: () => queryClient.getQueryData(["general_problem_solution"]) || null,
        staleTime: Infinity,
    });

    const { data: screenshots = [], refetch: refetchScreenshots } = useQuery<Screenshot[]>({
        queryKey: ["general_problem_screenshots"], 
        queryFn: fetchGeneralScreenshots,
        staleTime: 5 * 60 * 1000,
    });
    
    useEffect(() => {
        const handleGeneralProblemStart = () => setIsLoading(true);
        const handleGeneralProblemSuccess = (data: any) => {
            setIsLoading(false);
            showToast("Solution Ready", "The step-by-step solution has been generated.", "success");
        };
        const handleGeneralProblemError = (errMsg: string) => {
            setIsLoading(false);
            showToast("Error", errMsg, "error");
        };
        const handleReset = () => {
            queryClient.setQueryData(["general_problem_solution"], null);
            setView("queue");
        };
        const handleScreenshotTaken = () => refetchScreenshots();

        const cleanupFunctions = [
            window.electronAPI.onGeneralProblemStart(handleGeneralProblemStart),
            window.electronAPI.onGeneralProblemSuccess(handleGeneralProblemSuccess),
            window.electronAPI.onGeneralProblemError(handleGeneralProblemError),
            window.electronAPI.onResetView(handleReset),
            window.electronAPI.onScreenshotTaken(handleScreenshotTaken),
        ];
        return () => cleanupFunctions.forEach(fn => fn());
    }, [showToast, setView, queryClient, refetchScreenshots]);

    useEffect(() => {
        if (!contentRef.current) return;
        const updateDimensions = () => {
            if (!contentRef.current) return;
            const height = contentRef.current.scrollHeight || 600;
            const width = contentRef.current.scrollWidth || 800;
            window.electronAPI?.updateContentDimensions({ width, height });
        };
        updateDimensions();
        const resizeObserver = new ResizeObserver(updateDimensions);
        if (contentRef.current) resizeObserver.observe(contentRef.current);
        return () => resizeObserver.disconnect();
    }, [solutionData, solutionError, isLoading, screenshots]);

    const handleBackToQueue = () => {
        queryClient.setQueryData(["general_problem_solution"], null);
        setView("queue");
    };

    const handleDeleteScreenshot = async (index: number) => {
        if (!screenshots[index]) return;
        const pathToDelete = screenshots[index].path;
        try {
            await window.electronAPI.deleteScreenshot(pathToDelete);
            refetchScreenshots(); 
            showToast("Screenshot Deleted", "The screenshot has been removed.", "success");
        } catch (err: any) {
            console.error("Error deleting screenshot:", err);
            showToast("Error", err.message || "Could not delete screenshot.", "error");
        }
    };

    const getSolutionHtml = (): { __html: string } | undefined => { 
        if (!solutionData?.solution) return undefined; 
        try {
            // --- START OF MODIFIED CODE ---
            // Use the customMarked instance for parsing
            const rawHtml = customMarked.parse(solutionData.solution, { breaks: true, gfm: true }) as string;
            // --- END OF MODIFIED CODE ---
            return { __html: rawHtml }; 
        } catch (e: any) {
            console.error("Markdown parsing error (with KaTeX):", e);
            let errorText = solutionData.solution.replace(/</g, "&lt;").replace(/>/g, "&gt;");
            if (e.message) {
                errorText += `\n\n--- PARSING ERROR ---\n${e.message}`;
            }
            return { __html: `<pre>${errorText}</pre>` };
        }
    };

    return (
        <div ref={contentRef} className="relative p-4 space-y-4 bg-black/80 text-white rounded-lg w-full">
            <div className="flex justify-between items-center mb-4">
                <h1 className="text-2xl font-semibold text-white/90">General Problem Solver</h1>
                <button 
                    onClick={handleBackToQueue}
                    className="bg-gray-700 hover:bg-gray-600 text-white/90 font-medium py-2 px-4 rounded text-xs transition-colors"
                >
                    Back to Queue
                </button>
            </div>

            <div className="mb-6">
                <h2 className="text-lg font-medium text-white/80 mb-2">Problem Screenshot(s):</h2>
                {screenshots.length > 0 ? (
                    <ScreenshotQueue 
                        screenshots={screenshots.map(s => ({ path: s.path, preview: s.thumbnail }))} 
                        onDeleteScreenshot={handleDeleteScreenshot} 
                        isLoading={isLoading} 
                    />
                ) : (
                    <p className="text-gray-400 text-sm">
                        No screenshots submitted. Please go to the Queue to add screenshots.
                    </p>
                )}
            </div>

            {isLoading && (
                <div className="flex flex-col items-center justify-center py-10">
                    <div className="w-10 h-10 border-4 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                    <p className="ml-3 text-lg text-white/80 mt-3">Solving problem step-by-step...</p>
                </div>
            )}

            {solutionError && (
                 <div className="bg-red-900/30 border border-red-700 text-red-300 p-4 rounded-md">
                    <h3 className="font-semibold mb-1">Error Solving Problem:</h3>
                    <pre className="whitespace-pre-wrap text-sm">
                        {solutionError instanceof Error ? solutionError.message : String(solutionError)}
                    </pre>
                </div>
            )}
            
            {solutionData?.solution && !isLoading && (
                <div>
                    <h2 className="text-xl font-medium text-white/80 mb-3">Step-by-Step Solution:</h2>
                    <div 
                        className="prose prose-sm prose-invert max-w-none bg-black/60 border border-gray-700 p-4 rounded-md max-h-[80vh] overflow-y-auto"
                        dangerouslySetInnerHTML={getSolutionHtml()} 
                    />
                </div>
            )}
             {!solutionData?.solution && !isLoading && !solutionError && screenshots.length > 0 && (
                <p className="text-center text-gray-400 py-8">
                    Ready to solve. If you've added screenshots, the "Solve General" button in the Queue tab will process them.
                </p>
            )}
        </div>
    );
};

export default GeneralProblemSolver;