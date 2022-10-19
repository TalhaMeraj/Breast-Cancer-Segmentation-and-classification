function [Selection , SelectionValue]=psoSelection(Data,nOfSelection,population,itrationMax,fun)
% PSO SELECTION ALGORITHM
% INPUTS
% Data=Data to be minimize selection 
% nOfSelection=number of selection
% population=number of population
% itrationMax=maximum itration
% fun=fitness function
% OUTPUTS
% Selection=Selected Positions
% SelectionValue=fitness value of selected
% Developer Er.Abbas Manthiri S
% EMail abbasmanthiribe@gmail.com
% Date 23-03-2017

% Boundary
cloumn=size(Data,2);
if cloumn<=nOfSelection
    error('Number of Selection must be positive and less than Selection boundary')
end
Boundary=[1,cloumn];
% Initialize Constants
weight=0.2;
constant1=2;
constant2=2;
% Initialize Pariticles
Positions=zeros(population,nOfSelection);
for i=1:population
    tempVar = randperm(Boundary(2));
    Positions(i,:) = tempVar(1:nOfSelection);
end
% velocity initialization
Velocity = ones(population,nOfSelection);
% Initialize Global best
itration = 1;
globalBest.value(itration) = inf;
% fun=@distkm;
%% Optimizing
Particles = struct;

while(1)
    % Calculating Fitness Values
    for i=1:population
        Particles(itration).position(i,:) = Positions(i,:);
        Particles(itration).value(i) = fun(Data(:,Positions(i,:)));
    end
    % Update Position Best
    [Particlesbest.value(itration),ind] = min(Particles(itration).value);
    Particlesbest.position(itration,:) = Particles(itration).position(ind,:);
    % Update Global Best
    if globalBest.value(itration) > Particlesbest.value(itration)
        [globalBest.value(itration),globalBest.value(itration+1)] = deal(Particlesbest.value(itration));
        globalBest.position(itration,:) = Positions(ind(1),:);
    else
        globalBest.value(itration+1) = globalBest.value(itration);
        globalBest.position(itration,:) = globalBest.position(itration-1,:);
    end
    fprintf('\nIteration %d gbest value of  fitness= %f',itration,globalBest.value(itration))
    % Velocity Update
    Velocity = (weight*Velocity) + (constant1*rand(1)*(repmat(Particlesbest.position(itration,:),population,1)-Positions)) + ...
        (constant2*rand(1)*(repmat(globalBest.position(itration,:),population,1)-Positions));
    % Postion Update
    Positions = Positions+round(Velocity);
    % Boundary Checking for Position
    Positions(Positions>Boundary(2)) = round(rand(1)*(Boundary(2)-1))+1;
    Positions(Positions<Boundary(1)) = round(rand(1)*(Boundary(2)-1))+1;
    for i=1:population
        if length(unique( Positions(i,:) )) ~= nOfSelection
        tempVar = randperm(Boundary(2));
        Positions(i,:) = tempVar(1:nOfSelection);
        end
    end
    % Loop Breaking is Positions all are same or itertion acheived maximum
    % itertion
    count=0;
    for i=1:population-1;
        if Positions(i,:) == Positions(i+1,:)
            count = count+1;
        end
    end
    if (count == population-1) || (itration >= itrationMax)
        fprintf('\n******Iteration completed******\n')
        break
    end
    itration = itration+1;
end
Selection=globalBest.position(itration,:);
SelectionValue=globalBest.value(itration);
